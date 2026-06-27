@file:Suppress("UNCHECKED_CAST")

package com.wsr.knist.network

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.launch
import com.wsr.knist.core.unwrap
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.Process
import kotlinx.serialization.Serializable
import okio.BufferedSink
import okio.BufferedSource

private typealias TrainLambda = IOScope.(input: Batch<IOType>, context: Context) -> Batch<IOType>

@Serializable(with = NetworkSerializer::class)
class Network<I, O> internal constructor(
    val inputConverter: Converter<I>,
    val outputConverter: Converter<O>,
    val layers: List<Process>,
    val output: Output,
) {
    /**
     * 推論用の関数
     * @param モデルへの入力
     * @return モデルの出力
     */
    fun expect(input: I): O {
        val encoded = inputConverter._encode(input)
        val context = Context(input = encoded)
        val result = IOScope.launch {
            layers
                .fold(encoded) { acc, process -> with(process) { _expect(acc, context) } }
                .let { with(output) { _expect(it) } }
        }
        return outputConverter._decode(result)
    }

    /**
     * loss計算用の関数（逆伝播なし）
     * @param モデルへの入力
     * @return 損失関数の値
     */
    fun loss(input: I, label: O): Float = _loss(input) {
        outputConverter._encode(label)
    }

    inline fun loss(input: I, crossinline label: (O) -> O): Float = _loss(input) {
        val decoded = outputConverter._decode(it)
        outputConverter._encode(label(decoded))
    }

    @Suppress("FunctionName")
    @PublishedApi
    internal inline fun _loss(input: I, crossinline label: (Batch<IOType>) -> Batch<IOType>): Float {
        val encoded = inputConverter._encode(input)
        val context = Context(input = encoded)
        val result = IOScope.launch {
            val out = layers
                .fold(encoded) { acc, process -> with(process) { _expect(acc, context) } }
                .let { i -> with(output) { _train(i, { label(it) }) } }
            out.loss
        }
        return result.unwrap()
    }

    /**
     * 訓練用の関数
     * @param モデルへの入力
     * @return 損失関数の値
     */
    fun train(input: I, label: O): Float = _train(input) {
        outputConverter._encode(label)
    }

    inline fun train(input: I, crossinline label: (O) -> O): Float = _train(input) {
        val decoded = outputConverter._decode(it)
        val labeled = label(decoded)
        outputConverter._encode(labeled)
    }

    @Suppress("FunctionName")
    @PublishedApi
    internal inline fun _train(input: I, crossinline label: (Batch<IOType>) -> Batch<IOType>): Float {
        var loss = 0f
        val outputFn: TrainLambda = { encoded: Batch<IOType>, context: Context ->
            val out = with(output) { _train(encoded) { label(it) } }
            loss = out.loss.unwrap()
            out.delta
        }
        IOScope.launch {
            val encoded = inputConverter._encode(input)
            val context = Context(input = encoded)
            trainLambda(outputFn).invoke(this, encoded, context)
        }
        return loss
    }

    @PublishedApi
    internal val trainLambda: (TrainLambda) -> TrainLambda = run {
        val initial: (TrainLambda) -> TrainLambda = { it }
        layers.foldRight(initial) { layer: Process, acc: (TrainLambda) -> TrainLambda ->
            { final: TrainLambda ->
                { input: Batch<IOType>, context: Context ->
                    val scope = this
                    with(layer) {
                        scope._train(input, context) { i -> acc(final)(i, context) }
                    }
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

    fun clone(): Network<I, O> = NetworkSerializer.decodeFromCbor(NetworkSerializer.encodeToCbor(this))

    companion object {
        fun <I, O> fromJson(value: String) = NetworkSerializer.decodeFromString<I, O>(value)

        fun <I, O> fromJson(source: BufferedSource) = NetworkSerializer.decodeFromBufferedSource<I, O>(source)

        fun <I, O> fromCbor(bytes: ByteArray): Network<I, O> = NetworkSerializer.decodeFromCbor(bytes)

        fun <I, O> fromCbor(source: BufferedSource): Network<I, O> = NetworkSerializer.decodeFromCborSource(source)
    }
}
