package com.wsr.knist.network

import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.launch
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.process.Compute
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

private typealias TrainLambdaV2 = IOScope.(GraphEnv) -> Unit

@Serializable(with = NetworkSerializer::class)
class Network<I, O> @PublishedApi internal constructor(
    @PublishedApi internal val source: Graph.Source<I>,
    @PublishedApi internal val graph: List<Graph.Node>,
    @PublishedApi internal val sink: Graph.Sink<O>,
    @PublishedApi internal val optimizer: Optimizer,
    @PublishedApi internal val initializer: WeightInitializer,
) {
    @PublishedApi
    internal val mutex = Mutex()

    private val env = GraphEnv()

    suspend fun expect(input: I, dispatcher: CoroutineDispatcher = Dispatchers.Default): O = withContext(dispatcher) {
        mutex.withLock {
            env[source.id] = source.converter._encode(input)
            val output = IOScope.launch {
                val scope = this
                graph.forEach { node ->
                    when (node) {
                        is Graph.Node.Attach -> {
                            with(node.process) {
                                env[node.id] = scope._expect(env[node.from], env)
                            }
                        }

                        is Graph.Node.Connect -> {
                            with(node.join) {
                                env[node.id] = scope._expect(node.from.map { env[it] }, env)
                            }
                        }

                        is Graph.Node.Observe -> {
                            env[node.id] = env[node.from]
                        }
                    }
                }
                with(sink.output) { scope._expect(env[sink.from]) }
            }
            sink.converter._decode(output)
        }
    }

    suspend fun loss(input: I, label: O, dispatcher: CoroutineDispatcher = Dispatchers.Default): IOType.D0.Global =
        withContext(dispatcher) {
            mutex.withLock {
                env[source.id] = source.converter._encode(input)
                IOScope.launch {
                    val scope = this
                    graph.forEach { node ->
                        when (node) {
                            is Graph.Node.Attach -> {
                                with(node.process) {
                                    env[node.id] = scope._expect(env[node.from], env)
                                }
                            }

                            is Graph.Node.Connect -> {
                                with(node.join) {
                                    env[node.id] = scope._expect(node.from.map { env[it] }, env)
                                }
                            }

                            is Graph.Node.Observe -> {
                                env[node.id] = env[node.from]
                            }
                        }
                    }
                    val result = with(sink.output) {
                        scope._train(
                            input = env[sink.from],
                            label = { sink.converter._encode(label) },
                        )
                    }
                    result.loss.toGlobal()
                }
            }
        }

    suspend fun train(input: I, label: O, dispatcher: CoroutineDispatcher = Dispatchers.Default): IOType.D0.Global =
        withContext(dispatcher) {
            mutex.withLock {
                env[source.id] = source.converter._encode(input)
                IOScope.launch {
                    var loss: IOType.D0.Global? = null
                    val sinkStep: TrainLambdaV2 = {
                        val result = with(sink.output) {
                            _train(env[sink.from]) {
                                env.reset()
                                sink.converter._encode(label)
                            }
                        }
                        loss = result.loss.toGlobal()
                        env.plus(sink.from, result.delta)
                    }
                    trainLambda(sinkStep)(env)
                    loss!!
                }
            }
        }

    private val trainLambda: (TrainLambdaV2) -> TrainLambdaV2 = run {
        val initial: (TrainLambdaV2) -> TrainLambdaV2 = { it }
        graph.foldRight(initial) { node, next ->
            { final ->
                when (node) {
                    is Graph.Node.Attach -> {
                        { env ->
                            val delta = with(node.process) {
                                _train(env[node.from], env) { out ->
                                    env[node.id] = out
                                    next(final)(env)
                                    env[node.id]
                                }
                            }
                            env.plus(node.from, delta)
                        }
                    }

                    is Graph.Node.Connect -> {
                        { env ->
                            val delta = with(node.join) {
                                _train(node.from.map { env[it] }, env) { out ->
                                    env[node.id] = out
                                    next(final)(env)
                                    env[node.id]
                                }
                            }
                            node.from.forEachIndexed { index, id -> env.plus(id, delta[index]) }
                        }
                    }

                    is Graph.Node.Observe -> {
                        { env ->
                            env[node.id] = env[node.from]
                            next(final)(env)
                        }
                    }
                }
            }
        }
    }

    @JvmName("replaceOptimizer")
    fun replace(condition: (Process) -> Boolean, optimizer: Optimizer): Network<I, O> = clone().also { copy ->
        copy.graph.forEach { node ->
            if (node is Graph.Node.Attach && condition(node.process)) {
                node.process.update(optimizer)
            }
        }
    }

    fun <I2> replaceSource(converter: Converter<I2>): Network<I2, O> {
        val copy = clone()
        return Network(
            source = Graph.Source(id = copy.source.id, converter = converter),
            graph = copy.graph,
            sink = copy.sink,
            optimizer = copy.optimizer,
            initializer = copy.initializer,
        )
    }

    @JvmName("replaceSinkD1")
    fun <T : Output.D1, O2> replaceSink(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        block: GraphBuilder.Node.D1.() -> GraphBuilder.Result<O2>,
    ): Network<I, O2> {
        val copy = clone()
        val last = copy.graph.last() as Graph.Node.Attach
        check(last.process.outputShape.size == 1) {
            "invalid replaceSink. outputShape=${last.process.outputShape}"
        }
        val builder = GraphBuilder.Node.D1(
            inputI = last.process.outputShape[0],
            from = copy.sink.from,
            nodes = copy.graph,
            optimizer = optimizer,
            initializer = initializer,
        )
        val result = builder.block()
        return Network(
            source = copy.source,
            graph = result.nodes,
            sink = result.sink,
            optimizer = optimizer,
            initializer = initializer,
        )
    }

    @JvmName("replaceSinkD2")
    fun <T : Output.D2, O2> replaceSink(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        block: GraphBuilder.Node.D2.() -> GraphBuilder.Result<O2>,
    ): Network<I, O2> {
        val copy = clone()
        val last = copy.graph.last() as Graph.Node.Attach
        check(last.process.outputShape.size == 2) {
            "invalid replaceSink. outputShape=${last.process.outputShape}"
        }
        val builder = GraphBuilder.Node.D2(
            inputI = last.process.outputShape[0],
            inputJ = last.process.outputShape[1],
            from = copy.sink.from,
            nodes = copy.graph,
            optimizer = optimizer,
            initializer = initializer,
        )
        val result = builder.block()
        return Network(
            source = copy.source,
            graph = result.nodes,
            sink = result.sink,
            optimizer = optimizer,
            initializer = initializer,
        )
    }

    @JvmName("replaceComputeD1")
    inline fun <reified T : Compute.D1> replace(
        optimizer: Optimizer = this.optimizer,
        initializer: WeightInitializer = this.initializer,
        crossinline condition: (T) -> Boolean,
        crossinline block: GraphBuilder.Node.D1.(T) -> GraphBuilder.Node.D1,
    ): Network<I, O> = replace<T, GraphBuilder.Node.D1, GraphBuilder.Node.D1>(
        seed = { layer, from ->
            GraphBuilder.Node.D1(
                inputI = layer.inputI,
                from = from,
                nodes = emptyList(),
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
        crossinline block: GraphBuilder.Node.D2.(T) -> GraphBuilder.Node.D2,
    ): Network<I, O> = replace<T, GraphBuilder.Node.D2, GraphBuilder.Node.D2>(
        seed = { layer, from ->
            GraphBuilder.Node.D2(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                from = from,
                nodes = emptyList(),
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
        crossinline block: GraphBuilder.Node.D3.(T) -> GraphBuilder.Node.D3,
    ): Network<I, O> = replace<T, GraphBuilder.Node.D3, GraphBuilder.Node.D3>(
        seed = { layer, from ->
            GraphBuilder.Node.D3(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                inputK = layer.inputK,
                from = from,
                nodes = emptyList(),
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
        crossinline block: GraphBuilder.Node.D1.(T) -> GraphBuilder.Node.D2,
    ): Network<I, O> = replace<T, GraphBuilder.Node.D1, GraphBuilder.Node.D2>(
        seed = { layer, from ->
            GraphBuilder.Node.D1(
                inputI = layer.inputI,
                from = from,
                nodes = emptyList(),
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
        crossinline block: GraphBuilder.Node.D1.(T) -> GraphBuilder.Node.D3,
    ): Network<I, O> = replace<T, GraphBuilder.Node.D1, GraphBuilder.Node.D3>(
        seed = { layer, from ->
            GraphBuilder.Node.D1(
                inputI = layer.inputI,
                from = from,
                nodes = emptyList(),
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
        crossinline block: GraphBuilder.Node.D2.(T) -> GraphBuilder.Node.D1,
    ): Network<I, O> = replace<T, GraphBuilder.Node.D2, GraphBuilder.Node.D1>(
        seed = { layer, from ->
            GraphBuilder.Node.D2(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                from = from,
                nodes = emptyList(),
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
        crossinline block: GraphBuilder.Node.D2.(T) -> GraphBuilder.Node.D3,
    ): Network<I, O> = replace<T, GraphBuilder.Node.D2, GraphBuilder.Node.D3>(
        seed = { layer, from ->
            GraphBuilder.Node.D2(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                from = from,
                nodes = emptyList(),
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
        crossinline block: GraphBuilder.Node.D3.(T) -> GraphBuilder.Node.D1,
    ): Network<I, O> = replace<T, GraphBuilder.Node.D3, GraphBuilder.Node.D1>(
        seed = { layer, from ->
            GraphBuilder.Node.D3(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                inputK = layer.inputK,
                from = from,
                nodes = emptyList(),
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
        crossinline block: GraphBuilder.Node.D3.(T) -> GraphBuilder.Node.D2,
    ): Network<I, O> = replace<T, GraphBuilder.Node.D3, GraphBuilder.Node.D2>(
        seed = { layer, from ->
            GraphBuilder.Node.D3(
                inputI = layer.inputI,
                inputJ = layer.inputJ,
                inputK = layer.inputK,
                from = from,
                nodes = emptyList(),
                optimizer = optimizer,
                initializer = initializer,
            )
        },
        condition = condition,
        block = block,
    )

    @PublishedApi
    internal inline fun <reified T : Process, B1 : GraphBuilder.Node, B2 : GraphBuilder.Node> replace(
        crossinline seed: (layer: T, from: GraphId) -> B1,
        crossinline condition: (T) -> Boolean,
        crossinline block: B1.(T) -> B2,
    ): Network<I, O> {
        val copy = clone()
        val redirect = mutableMapOf<GraphId, GraphId>()
        val newNodes = copy.graph.fold(emptyList<Graph.Node>()) { nodes, node ->
            when (node) {
                is Graph.Node.Attach -> {
                    val process = node.process
                    if (process !is T || !condition(process)) {
                        return@fold nodes + Graph.Node.Attach(
                            id = node.id,
                            from = redirect[node.from] ?: node.from,
                            process = process,
                        )
                    }

                    val from = redirect[node.from] ?: node.from
                    val result = seed(process, from).block(process)
                    if (result.nodes.isEmpty()) {
                        check(process.inputShape == process.outputShape) {
                            """
                                invalid replace.
                                input: ${process.inputShape}
                                output: ${process.outputShape}
                            """.trimIndent()
                        }
                        redirect[node.id] = from
                        return@fold nodes
                    }

                    val first = (result.nodes.first() as Graph.Node.Attach).process
                    val last = (result.nodes.last() as Graph.Node.Attach).process
                    check(first.inputShape == process.inputShape && last.outputShape == process.outputShape) {
                        """
                            invalid replace.
                            input: ${process.inputShape}
                            output: ${process.outputShape}
                            replaced input: ${first.inputShape}
                            replaced output: ${last.outputShape}
                        """.trimIndent()
                    }
                    redirect[node.id] = result.from
                    nodes + result.nodes
                }

                is Graph.Node.Connect -> nodes + Graph.Node.Connect(
                    id = node.id,
                    from = node.from.map { redirect[it] ?: it },
                    join = node.join,
                )

                is Graph.Node.Observe -> nodes + Graph.Node.Observe(
                    id = node.id,
                    from = redirect[node.from] ?: node.from,
                )
            }
        }
        return Network(
            source = copy.source,
            graph = newNodes,
            sink = Graph.Sink(
                id = copy.sink.id,
                from = redirect[copy.sink.from] ?: copy.sink.from,
                output = copy.sink.output,
                converter = copy.sink.converter,
            ),
            optimizer = copy.optimizer,
            initializer = copy.initializer,
        )
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

abstract class GraphNetwork<T: GraphNetwork<T>> {
    abstract val sources: List<Graph.Source<*>>
    abstract val graph: List<Graph.Node>
    abstract val sinks: List<Graph.Sink<*>>

    abstract val optimizer: Optimizer
    abstract val initializer: WeightInitializer

    abstract fun create(
        sources: List<Graph.Source<*>>,
        graph: List<Graph.Node>,
        sinks: List<Graph.Sink<*>>,
        optimizer: Optimizer,
        initializer: WeightInitializer,
    ): T

    abstract fun clone(): T
}
