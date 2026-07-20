package com.wsr.knist.network

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.launch
import com.wsr.knist.network.process.Context
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import okio.BufferedSink
import okio.BufferedSource

private typealias TrainLambdaV2 = IOScope.(Context) -> Unit

@Serializable(with = NetworkV2Serializer::class)
class NetworkV2<I, O> @PublishedApi internal constructor(
    @PublishedApi internal val source: Graph.Source<I>,
    @PublishedApi internal val graph: List<Graph.Node>,
    @PublishedApi internal val sink: Graph.Sink<O>,
) {
    @PublishedApi
    internal val mutex = Mutex()

    private val env = mutableMapOf<GraphId, Batch<IOType>>()

    suspend fun expect(input: I, dispatcher: CoroutineDispatcher = Dispatchers.Default): O = withContext(dispatcher) {
        mutex.withLock {
            val input = source.converter._encode(input).also { env[source.id] = it }
            val context = Context(input)
            val output = IOScope.launch {
                val scope = this
                graph.forEach { node ->
                    when (node) {
                        is Graph.Node.Attach -> {
                            with(node.process) {
                                env[node.id] = scope._expect(env[node.from]!!, context)
                            }
                        }
                    }
                }
                with(sink.output) { scope._expect(env[sink.from]!!) }
            }
            sink.converter._decode(output)
        }
    }

    suspend fun loss(input: I, label: O, dispatcher: CoroutineDispatcher = Dispatchers.Default): IOType.D0.Global =
        withContext(dispatcher) {
            mutex.withLock {
                val input = source.converter._encode(input).also { env[source.id] = it }
                val context = Context(input)
                IOScope.launch {
                    val scope = this
                    graph.forEach { node ->
                        when (node) {
                            is Graph.Node.Attach -> {
                                with(node.process) {
                                    env[node.id] = scope._expect(env[node.from]!!, context)
                                }
                            }
                        }
                    }
                    val result = with(sink.output) {
                        scope._train(
                            input = env[sink.from]!!,
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
                val input = source.converter._encode(input).also { env[source.id] = it }
                val context = Context(input)
                IOScope.launch {
                    var loss: IOType.D0.Global? = null
                    val sinkStep: TrainLambdaV2 = {
                        val result = with(sink.output) {
                            _train(env[sink.from]!!) { sink.converter._encode(label) }
                        }
                        loss = result.loss.toGlobal()
                        env[sink.from] = result.delta
                    }
                    trainLambda(sinkStep)(context)
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
                        { context ->
                            val delta = with(node.process) {
                                _train(env[node.from]!!, context) { out ->
                                    env[node.id] = out
                                    next(final)(context)
                                    env[node.id]!!
                                }
                            }
                            env[node.from] = delta
                        }
                    }
                }
            }
        }
    }

    fun toJson(): String = NetworkV2Serializer.encodeToString(this)

    fun toJson(sink: BufferedSink) {
        NetworkV2Serializer.encodeToBufferedSink(
            value = this,
            sink = sink,
        )
    }

    fun toCbor(): ByteArray = NetworkV2Serializer.encodeToCbor(this)

    fun toCbor(sink: BufferedSink) = NetworkV2Serializer.encodeToCborSink(this, sink)

    fun clone(): NetworkV2<I, O> = NetworkV2Serializer.decodeFromCbor(NetworkV2Serializer.encodeToCbor(this))

    companion object {
        fun <I, O> fromJson(value: String) = NetworkV2Serializer.decodeFromString<I, O>(value)

        fun <I, O> fromJson(source: BufferedSource) = NetworkV2Serializer.decodeFromBufferedSource<I, O>(source)

        fun <I, O> fromCbor(bytes: ByteArray): NetworkV2<I, O> = NetworkV2Serializer.decodeFromCbor(bytes)

        fun <I, O> fromCbor(source: BufferedSource): NetworkV2<I, O> = NetworkV2Serializer.decodeFromCborSource(source)
    }
}
