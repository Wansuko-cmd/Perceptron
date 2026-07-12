@file:Suppress("UNCHECKED_CAST")

package com.wsr.knist.network

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.launch
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.process.Compute
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.Process
import com.wsr.knist.network.process.Reshape
import kotlin.jvm.JvmName
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import okio.BufferedSink
import okio.BufferedSource

private typealias TrainLambda = IOScope.(input: Batch<IOType>, context: Context) -> Batch<IOType>

@Serializable(with = NetworkSerializer::class)
class Network<I, O> @PublishedApi internal constructor(
    @PublishedApi internal val inputConverter: Converter<I>,
    @PublishedApi internal val outputConverter: Converter<O>,
    @PublishedApi internal val layers: List<Process>,
    @PublishedApi internal val output: Output,
    @PublishedApi internal val optimizer: Optimizer,
    @PublishedApi internal val initializer: WeightInitializer,
) {
    @PublishedApi
    internal val mutex = Mutex()

    /**
     * 推論用の関数
     * @param モデルへの入力
     * @return モデルの出力
     */
    suspend fun expect(input: I, dispatcher: CoroutineDispatcher = Dispatchers.Default): O = withContext(dispatcher) {
        mutex.withLock {
            val encoded = inputConverter._encode(input)
            val context = Context(input = encoded)
            val result = IOScope.launch {
                layers
                    .fold(encoded) { acc, process -> with(process) { _expect(acc, context) } }
                    .let { with(output) { _expect(it) } }
            }
            outputConverter._decode(result)
        }
    }

    /**
     * loss計算用の関数（逆伝播なし）
     * @param モデルへの入力
     * @return 損失関数の値
     */
    suspend fun loss(input: I, label: O, dispatcher: CoroutineDispatcher = Dispatchers.Default): IOType.D0.Global =
        withContext(dispatcher) {
            mutex.withLock {
                _loss(input) {
                    outputConverter._encode(label)
                }
            }
        }

    suspend inline fun loss(
        input: I,
        dispatcher: CoroutineDispatcher = Dispatchers.Default,
        crossinline label: (O) -> O,
    ): IOType.D0.Global = withContext(dispatcher) {
        mutex.withLock {
            _loss(input) {
                val decoded = outputConverter._decode(it)
                outputConverter._encode(label(decoded))
            }
        }
    }

    @Suppress("FunctionName")
    @PublishedApi
    internal inline fun _loss(input: I, crossinline label: (Batch<IOType>) -> Batch<IOType>): IOType.D0.Global {
        val encoded = inputConverter._encode(input)
        val context = Context(input = encoded)
        return IOScope.launch {
            val out = layers
                .fold(encoded) { acc, process -> with(process) { _expect(acc, context) } }
                .let { i -> with(output) { _train(i, { label(it) }) } }
            out.loss.toGlobal()
        }
    }

    /**
     * 訓練用の関数
     * @param モデルへの入力
     * @return 損失関数の値
     */
    suspend fun train(input: I, label: O, dispatcher: CoroutineDispatcher = Dispatchers.Default): IOType.D0.Global =
        withContext(dispatcher) {
            mutex.withLock {
                _train(input) {
                    outputConverter._encode(label)
                }
            }
        }
    suspend inline fun train(
        input: I,
        dispatcher: CoroutineDispatcher = Dispatchers.Default,
        crossinline label: (O) -> O,
    ): IOType.D0.Global = withContext(dispatcher) {
        mutex.withLock {
            _train(input) {
                val decoded = outputConverter._decode(it)
                val labeled = label(decoded)
                outputConverter._encode(labeled)
            }
        }
    }

    @Suppress("FunctionName")
    @PublishedApi
    internal inline fun _train(input: I, crossinline label: (Batch<IOType>) -> Batch<IOType>): IOType.D0.Global {
        var loss: IOType.D0.Global? = null
        val outputFn: TrainLambda = { encoded: Batch<IOType>, context: Context ->
            val out = with(output) { _train(encoded) { label(it) } }
            loss = out.loss.toGlobal()
            out.delta
        }
        IOScope.launch {
            val encoded = inputConverter._encode(input)
            val context = Context(input = encoded)
            trainLambda(outputFn).invoke(this, encoded, context)
        }
        return loss!!
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

    @JvmName("replaceOptimizer")
    fun replace(condition: (Process) -> Boolean, optimizer: Optimizer): Network<I, O> = clone().also { copy ->
        copy.layers.forEach { layer -> if (condition(layer)) layer.update(optimizer) }
    }

    @JvmName("replaceComputeD1")
    inline fun <reified T : Compute.D1> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        crossinline condition: (T) -> Boolean,
        crossinline block: NetworkBuilder.D1<I>.(T) -> NetworkBuilder.D1<I>,
    ): Network<I, O> = replace<T, NetworkBuilder.D1<I>, NetworkBuilder.D1<I>>(
        seed = { layer ->
            NetworkBuilder.D1(
                inputI = layer.inputI,
                input = inputConverter,
                layers = emptyList(),
                optimizer = optimizer,
                initializer = initializer,
            )
        },
        condition = condition,
        block = block,
    )

    @JvmName("replaceComputeD2")
    inline fun <reified T : Compute.D2> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        crossinline condition: (T) -> Boolean,
        crossinline block: NetworkBuilder.D2<I>.(T) -> NetworkBuilder.D2<I>,
    ): Network<I, O> = replace<T, NetworkBuilder.D2<I>, NetworkBuilder.D2<I>>(
        seed = { layer ->
            NetworkBuilder.D2(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                input = inputConverter,
                layers = emptyList(),
                optimizer = optimizer,
                initializer = initializer,
            )
        },
        condition = condition,
        block = block,
    )

    @JvmName("replaceComputeD3")
    inline fun <reified T : Compute.D3> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        crossinline condition: (T) -> Boolean,
        crossinline block: NetworkBuilder.D3<I>.(T) -> NetworkBuilder.D3<I>,
    ): Network<I, O> = replace<T, NetworkBuilder.D3<I>, NetworkBuilder.D3<I>>(
        seed = { layer ->
            NetworkBuilder.D3(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                inputK = layer.inputK,
                input = inputConverter,
                layers = emptyList(),
                optimizer = optimizer,
                initializer = initializer,
            )
        },
        condition = condition,
        block = block,
    )

    @JvmName("replaceReshapeD1ToD2")
    inline fun <reified T : Reshape.D1ToD2> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        crossinline condition: (T) -> Boolean,
        crossinline block: NetworkBuilder.D1<I>.(T) -> NetworkBuilder.D2<I>,
    ): Network<I, O> = replace<T, NetworkBuilder.D1<I>, NetworkBuilder.D2<I>>(
        seed = { layer ->
            NetworkBuilder.D1(
                inputI = layer.inputI,
                input = inputConverter,
                layers = emptyList(),
                optimizer = optimizer,
                initializer = initializer,
            )
        },
        condition = condition,
        block = block,
    )

    @JvmName("replaceReshapeD1ToD3")
    inline fun <reified T : Reshape.D1ToD3> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        crossinline condition: (T) -> Boolean,
        crossinline block: NetworkBuilder.D1<I>.(T) -> NetworkBuilder.D3<I>,
    ): Network<I, O> = replace<T, NetworkBuilder.D1<I>, NetworkBuilder.D3<I>>(
        seed = { layer ->
            NetworkBuilder.D1(
                inputI = layer.inputI,
                input = inputConverter,
                layers = emptyList(),
                optimizer = optimizer,
                initializer = initializer,
            )
        },
        condition = condition,
        block = block,
    )

    @JvmName("replaceReshapeD2ToD1")
    inline fun <reified T : Reshape.D2ToD1> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        crossinline condition: (T) -> Boolean,
        crossinline block: NetworkBuilder.D2<I>.(T) -> NetworkBuilder.D1<I>,
    ): Network<I, O> = replace<T, NetworkBuilder.D2<I>, NetworkBuilder.D1<I>>(
        seed = { layer ->
            NetworkBuilder.D2(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                input = inputConverter,
                layers = emptyList(),
                optimizer = optimizer,
                initializer = initializer,
            )
        },
        condition = condition,
        block = block,
    )

    @JvmName("replaceReshapeD2ToD3")
    inline fun <reified T : Reshape.D2ToD3> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        crossinline condition: (T) -> Boolean,
        crossinline block: NetworkBuilder.D2<I>.(T) -> NetworkBuilder.D3<I>,
    ): Network<I, O> = replace<T, NetworkBuilder.D2<I>, NetworkBuilder.D3<I>>(
        seed = { layer ->
            NetworkBuilder.D2(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                input = inputConverter,
                layers = emptyList(),
                optimizer = optimizer,
                initializer = initializer,
            )
        },
        condition = condition,
        block = block,
    )

    @JvmName("replaceReshapeD3ToD1")
    inline fun <reified T : Reshape.D3ToD1> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        crossinline condition: (T) -> Boolean,
        crossinline block: NetworkBuilder.D3<I>.(T) -> NetworkBuilder.D1<I>,
    ): Network<I, O> = replace<T, NetworkBuilder.D3<I>, NetworkBuilder.D1<I>>(
        seed = { layer ->
            NetworkBuilder.D3(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                inputK = layer.inputK,
                input = inputConverter,
                layers = emptyList(),
                optimizer = optimizer,
                initializer = initializer,
            )
        },
        condition = condition,
        block = block,
    )

    @JvmName("replaceReshapeD3ToD2")
    inline fun <reified T : Reshape.D3ToD2> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        crossinline condition: (T) -> Boolean,
        crossinline block: NetworkBuilder.D3<I>.(T) -> NetworkBuilder.D2<I>,
    ): Network<I, O> = replace<T, NetworkBuilder.D3<I>, NetworkBuilder.D2<I>>(
        seed = { layer ->
            NetworkBuilder.D3(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                inputK = layer.inputK,
                input = inputConverter,
                layers = emptyList(),
                optimizer = optimizer,
                initializer = initializer,
            )
        },
        condition = condition,
        block = block,
    )

    @PublishedApi
    internal inline fun <reified T : Process, B1 : NetworkBuilder<*, *>, B2 : NetworkBuilder<*, *>> replace(
        crossinline seed: (T) -> B1,
        crossinline condition: (T) -> Boolean,
        crossinline block: B1.(T) -> B2,
    ): Network<I, O> {
        val copy = clone()
        val layers = copy.layers
            .fold(listOf<Process>()) { acc, layer ->
                when {
                    layer !is T || !condition(layer) -> acc + layer

                    else -> {
                        val next = seed(layer).block(layer).layers
                        if (next.isEmpty()) {
                            check(layer.inputShape == layer.outputShape)
                            acc
                        } else {
                            check(layer.inputShape == next.first().inputShape)
                            check(layer.outputShape == next.last().outputShape)
                            acc + next
                        }
                    }
                }
            }
        return Network(
            inputConverter = copy.inputConverter,
            outputConverter = copy.outputConverter,
            layers = layers,
            output = copy.output,
            optimizer = copy.optimizer,
            initializer = copy.initializer,
        )
    }

    fun <I2> replaceInput(input: Converter<I2>): Network<I2, O> {
        val copy = clone()
        return Network(
            inputConverter = input,
            outputConverter = copy.outputConverter,
            layers = copy.layers,
            output = copy.output,
            optimizer = copy.optimizer,
            initializer = copy.initializer,
        )
    }

    fun <O2> replaceOutput(output: Converter<O2>): Network<I, O2> {
        val copy = clone()
        return Network(
            inputConverter = copy.inputConverter,
            outputConverter = output,
            layers = copy.layers,
            output = copy.output,
            optimizer = copy.optimizer,
            initializer = copy.initializer,
        )
    }

    @JvmName("replaceOutputD1")
    fun <T : Output.D1> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        block: NetworkBuilder.D1<I>.(Converter.D1<O>) -> Network<I, O>,
    ): Network<I, O> {
        val copy = clone()
        check(copy.layers.last().outputShape.size == 1)
        return NetworkBuilder.D1(
            inputI = copy.layers.last().outputShape[0],
            input = copy.inputConverter,
            layers = copy.layers,
            optimizer = optimizer,
            initializer = initializer,
        ).block(copy.outputConverter as Converter.D1<O>)
    }

    @JvmName("replaceOutputD2")
    fun <T : Output.D2> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        block: NetworkBuilder.D2<I>.(Converter.D2<O>) -> Network<I, O>,
    ): Network<I, O> {
        val copy = clone()
        check(copy.layers.last().outputShape.size == 2)
        return NetworkBuilder.D2(
            inputI = copy.layers.last().outputShape[0],
            inputJ = copy.layers.last().outputShape[1],
            input = copy.inputConverter,
            layers = copy.layers,
            optimizer = optimizer,
            initializer = initializer,
        ).block(copy.outputConverter as Converter.D2<O>)
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
