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
import kotlinx.serialization.Serializable
import okio.BufferedSink
import okio.BufferedSource

@Serializable(with = NetworkSerializer::class)
class Network<I, O> @PublishedApi internal constructor(
    @PublishedApi internal val source: Graph.Source<I>,
    override val graph: List<Graph.Node>,
    @PublishedApi internal val sink: Graph.Sink<O>,
    override val optimizer: Optimizer,
    override val initializer: WeightInitializer,
) : GraphNetwork<Network<I, O>>() {

    override val sinks: List<Graph.Sink<*>> = listOf(sink)
    override val sources: List<Graph.Source<*>> = listOf(source)

    suspend fun expect(input: I, dispatcher: CoroutineDispatcher = Dispatchers.Default): O {
        val inputs = listOf(source.converter._encode(input))
        val outputs = _expect(inputs = inputs, dispatcher = dispatcher)
        return sink.converter._decode(outputs[0])
    }

    suspend fun loss(input: I, label: O, dispatcher: CoroutineDispatcher = Dispatchers.Default): IOType.D0.Global {
        val inputs = listOf(source.converter._encode(input))
        val labels = listOf<(Batch<IOType>) -> Batch<IOType>> { sink.converter._encode(label) }
        return _loss(inputs = inputs, labels = labels, dispatcher = dispatcher)[0]
    }

    suspend fun train(input: I, label: O, dispatcher: CoroutineDispatcher = Dispatchers.Default): IOType.D0.Global {
        val inputs = listOf(source.converter._encode(input))
        val labels = listOf<(Batch<IOType>) -> Batch<IOType>> { sink.converter._encode(label) }
        return _train(inputs = inputs, labels = labels, dispatcher = dispatcher)[0]
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

    fun toJson(): String = NetworkSerializer.encodeToString(this)

    fun toJson(sink: BufferedSink) {
        NetworkSerializer.encodeToBufferedSink(
            value = this,
            sink = sink,
        )
    }

    fun toCbor(): ByteArray = NetworkSerializer.encodeToCbor(this)

    fun toCbor(sink: BufferedSink) = NetworkSerializer.encodeToCborSink(this, sink)

    @Suppress("UNCHECKED_CAST")
    override fun create(
        sources: List<Graph.Source<*>>,
        graph: List<Graph.Node>,
        sinks: List<Graph.Sink<*>>,
        optimizer: Optimizer,
        initializer: WeightInitializer,
    ): Network<I, O> = Network(
        source = sources[0] as Graph.Source<I>,
        graph = graph,
        sink = sinks[0] as Graph.Sink<O>,
        optimizer = optimizer,
        initializer = initializer,
    )

    override fun clone(): Network<I, O> = NetworkSerializer.decodeFromCbor(NetworkSerializer.encodeToCbor(this))

    companion object {
        fun <I, O> fromJson(value: String) = NetworkSerializer.decodeFromString<I, O>(value)

        fun <I, O> fromJson(source: BufferedSource) = NetworkSerializer.decodeFromBufferedSource<I, O>(source)

        fun <I, O> fromCbor(bytes: ByteArray): Network<I, O> = NetworkSerializer.decodeFromCbor(bytes)

        fun <I, O> fromCbor(source: BufferedSource): Network<I, O> = NetworkSerializer.decodeFromCborSource(source)
    }
}
