@file:OptIn(ExperimentalSerializationApi::class)

package com.wsr.knist.network

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.output.Output
import kotlin.jvm.JvmName
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.okio.decodeFromBufferedSource
import okio.BufferedSource

interface Network {
    interface Src1 {
        class Sink1<I, O>(
            val source: Graph.Source<I>,
            override val graph: List<Graph.Node>,
            val sink: Graph.Sink<O>,
            override val optimizer: Optimizer,
            override val initializer: WeightInitializer,
        ) : GraphNetwork<Sink1<I, O>>() {
            override val sinks: List<Graph.Sink<*>> = listOf(sink)
            override val sources: List<Graph.Source<*>> = listOf(source)

            suspend fun expect(input: I, dispatcher: CoroutineDispatcher = Dispatchers.Default): O = _expect(
                inputs = listOf(source.converter._encode(input)),
                dispatcher = dispatcher,
            ) { outputs -> sink.converter._decode(outputs[0]) }

            suspend fun loss(
                input: I,
                label: O,
                dispatcher: CoroutineDispatcher = Dispatchers.Default,
            ): IOType.D0.Global {
                val inputs = listOf(source.converter._encode(input))
                val labels = listOf<(Batch<IOType>) -> Batch<IOType>> { sink.converter._encode(label) }
                return _loss(inputs = inputs, labels = labels, dispatcher = dispatcher)[0]
            }

            suspend fun train(
                input: I,
                label: O,
                dispatcher: CoroutineDispatcher = Dispatchers.Default,
            ): IOType.D0.Global {
                val inputs = listOf(source.converter._encode(input))
                val labels = listOf<(Batch<IOType>) -> Batch<IOType>> { sink.converter._encode(label) }
                return _train(inputs = inputs, labels = labels, dispatcher = dispatcher)[0]
            }

            fun <I2> replaceSource(converter: Converter<I2>): Sink1<I2, O> {
                val copy = clone()
                return Sink1(
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
            ): Sink1<I, O2> {
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
                return Sink1(
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
            ): Sink1<I, O2> {
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
                return Sink1(
                    source = copy.source,
                    graph = result.nodes,
                    sink = result.sink,
                    optimizer = optimizer,
                    initializer = initializer,
                )
            }

            override fun create(
                sources: List<Graph.Source<*>>,
                graph: List<Graph.Node>,
                sinks: List<Graph.Sink<*>>,
                optimizer: Optimizer,
                initializer: WeightInitializer,
            ): Sink1<I, O> = build(sources, graph, sinks, optimizer, initializer)

            override fun serializer(): GraphNetworkSerializer<Sink1<I, O>> = serializer<I, O>()

            companion object {
                @Suppress("UNCHECKED_CAST")
                private fun <I, O> build(
                    sources: List<Graph.Source<*>>,
                    graph: List<Graph.Node>,
                    sinks: List<Graph.Sink<*>>,
                    optimizer: Optimizer,
                    initializer: WeightInitializer,
                ): Sink1<I, O> {
                    check(sources.size == 1) { "invalid Network format. sources.size=${sources.size}." }
                    check(sinks.size == 1) { "invalid Network format. sinks.size=${sinks.size}." }
                    return Sink1(
                        source = sources[0] as Graph.Source<I>,
                        graph = graph,
                        sink = sinks[0] as Graph.Sink<O>,
                        optimizer = optimizer,
                        initializer = initializer,
                    )
                }

                private fun <I, O> serializer(): GraphNetworkSerializer<Sink1<I, O>> = GraphNetworkSerializer(::build)

                fun <I, O> fromJson(value: String): Sink1<I, O> =
                    networkSerializerJson.decodeFromString(serializer(), value)

                fun <I, O> fromJson(source: BufferedSource): Sink1<I, O> =
                    networkSerializerJson.decodeFromBufferedSource(serializer(), source)

                fun <I, O> fromCbor(bytes: ByteArray): Sink1<I, O> =
                    networkSerializerCbor.decodeFromByteArray(serializer(), bytes)

                fun <I, O> fromCbor(source: BufferedSource): Sink1<I, O> =
                    networkSerializerCbor.decodeFromByteArray(serializer(), source.readByteArray())
            }
        }
    }
}
