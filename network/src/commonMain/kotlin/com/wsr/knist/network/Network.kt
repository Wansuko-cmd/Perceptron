@file:Suppress("UNCHECKED_CAST")

package com.wsr.knist.network

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.Process
import kotlinx.serialization.Serializable
import okio.BufferedSink
import okio.BufferedSource

private typealias TrainLambda = (input: Batch<IOType>, context: Context) -> Batch<IOType>

@Serializable(with = NetworkSerializer::class)
class Network<I, O> internal constructor(
    val inputConverter: Converter,
    val outputConverter: Converter,
    val layers: List<Process>,
    val output: Output,
) {
    /**
     * 推論用の関数
     * @param モデルへの入力
     * @return モデルの出力
     */
    fun expect(input: I): O = expect(input = listOf(input))[0]

    fun expect(input: List<I>): List<O> {
        val input = inputConverter._encode(input)
        val context = Context(input = input)
        val output = layers
            .fold(input) { acc, process -> process._expect(acc, context) }
            .let { output._expect(it) }
        return outputConverter._decode(output) as List<O>
    }

    /**
     * loss計算用の関数（逆伝播なし）
     * @param モデルへの入力
     * @return 損失関数の値
     */
    fun loss(input: I, label: O) = loss(input = listOf(input), label = listOf(label))

    fun loss(input: I, label: (O) -> O) = loss(input = listOf(input)) { listOf(label(it[0])) }

    fun loss(input: List<I>, label: List<O>): Float = _loss(input) {
        outputConverter._encode(label)
    }

    fun loss(input: List<I>, label: (List<O>) -> List<O>): Float = _loss(input) {
        val output = outputConverter._decode(it) as List<O>
        val label = label(output)
        outputConverter._encode(label)
    }

    @Suppress("FunctionName")
    private inline fun _loss(input: List<I>, crossinline label: (Batch<IOType>) -> Batch<IOType>): Float {
        val input = inputConverter._encode(input)
        val context = Context(input = input)
        val output = layers
            .fold(input) { acc, process -> process._expect(acc, context) }
            .let { output._train(it, { label(it) }) }
        return output.loss
    }

    /**
     * 訓練用の関数
     * @param モデルへの入力
     * @return 損失関数の値
     */
    fun train(input: I, label: O) = train(input = listOf(input), label = listOf(label))

    fun train(input: I, label: (O) -> O) = train(input = listOf(input)) { listOf(label(it[0])) }

    fun train(input: List<I>, label: List<O>): Float = _train(input) {
        outputConverter._encode(label)
    }

    fun train(input: List<I>, label: (List<O>) -> List<O>): Float = _train(input) {
        val output = outputConverter._decode(it) as List<O>
        val label = label(output)
        outputConverter._encode(label)
    }

    @Suppress("FunctionName")
    private inline fun _train(input: List<I>, crossinline label: (Batch<IOType>) -> Batch<IOType>): Float {
        var loss = 0f
        val output: TrainLambda = { input: Batch<IOType>, context: Context ->
            val output = output._train(input) { label(it) }
            loss = output.loss
            output.delta
        }
        val input = inputConverter._encode(input)
        val context = Context(input = input)
        trainLambda(output).invoke(input, context)
        return loss
    }

    private val trainLambda: (TrainLambda) -> TrainLambda = run {
        val initial: (TrainLambda) -> TrainLambda = { it }
        layers.foldRight(initial) { layer: Process, acc: (TrainLambda) -> TrainLambda ->
            { final: TrainLambda ->
                { input: Batch<IOType>, context: Context ->
                    layer._train(input, context) { i -> acc(final)(i, context) }
                }
            }
        }
    }

    fun toJson(): String = NetworkSerializer.encodeToString(this)

    fun toJson(sink: BufferedSink) {
        NetworkSerializer.encodeToBufferedSink(
            value = this,
            sink = sink,
        )
    }

    fun toCbor(): ByteArray = NetworkSerializer.encodeToCbor(this)

    fun toCbor(sink: BufferedSink) = NetworkSerializer.encodeToCborSink(this, sink)

    companion object {
        fun <I, O> fromJson(value: String) = NetworkSerializer.decodeFromString<I, O>(value)

        fun <I, O> fromJson(source: BufferedSource) = NetworkSerializer.decodeFromBufferedSource<I, O>(source)

        fun <I, O> fromCbor(bytes: ByteArray): Network<I, O> = NetworkSerializer.decodeFromCbor(bytes)

        fun <I, O> fromCbor(source: BufferedSource): Network<I, O> = NetworkSerializer.decodeFromCborSource(source)
    }
}
