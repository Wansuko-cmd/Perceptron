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

private typealias TrainLambdaV2 = IOScope.(Context) -> Unit

class NetworkV2<I, O>(
    private val source: Graph.Source<I>,
    private val graph: List<Graph.Node>,
    private val sink: Graph.Sink<O>,
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

    companion object
}
