package com.wsr.knist.network

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.launch
import com.wsr.knist.network.process.Context

private typealias TrainLambdaV2 = IOScope.() -> Unit

class NetworkV2<I, O>(
    private val source: Graph.Source<I>,
    private val graph: List<Graph.Node>,
    private val sink: Graph.Sink<O>,
) {
    fun expect(input: I): O {
        val env = mutableMapOf< GraphId, Batch<IOType>>()
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
        return sink.converter._decode(output)
    }

    fun train(input: I, label: O): IOType.D0.Global {
        val env = mutableMapOf<GraphId, Batch<IOType>>()
        val input = source.converter._encode(input).also { env[source.id] = it }
        val context = Context(input)
        return IOScope.launch {
            var loss: IOType.D0.Global? = null
            val sinkStep: TrainLambdaV2 = {
                val result = with(sink.output) {
                    _train(env[sink.from]!!) { sink.converter._encode(label) }
                }
                loss = result.loss.toGlobal()
                env[sink.from] = result.delta
            }
            val chainStep = graph.foldRight(sinkStep) { node, next ->
                when (node) {
                    is Graph.Node.Attach -> {
                        {
                            val delta = with(node.process) {
                                _train(env[node.from]!!, context) { out ->
                                    env[node.id] = out
                                    next()
                                    env[node.id]!!
                                }
                            }
                            env[node.from] = delta
                        }
                    }
                }
            }
            chainStep()
            loss!!
        }
    }

    companion object
}
