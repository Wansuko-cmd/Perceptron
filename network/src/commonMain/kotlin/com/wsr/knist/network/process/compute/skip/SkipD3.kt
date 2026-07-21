@file:Suppress("UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.skip

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.Graph
import com.wsr.knist.network.GraphBuilder
import com.wsr.knist.network.GraphScope.addCompute
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Compute
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.process.Process
import com.wsr.knist.network.process.Reshape
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

private typealias CALC_DELTA_D3 = IOScope.(input: Batch<IOType.D3>, env: GraphEnv) -> Batch<IOType.D3>

@Serializable
class SkipD3 internal constructor(
    // List<Process.D3>だがSerializer対策
    private val layers: List<Process> = emptyList(),
    override val inputI: Int,
    override val inputJ: Int,
    override val inputK: Int,
    override val id: String = Uuid.random().toString(),
) : Compute.D3() {
    override val outputI: Int get() = inputI
    override val outputJ: Int get() = inputJ
    override val outputK: Int get() = inputK
    override fun IOScope.expect(input: Batch<IOType.D3>, env: GraphEnv): Batch<IOType.D3> {
        val scope = this
        val main = layers.fold(input) { acc, layer ->
            with(layer) { scope._expect(acc, env) } as Batch<IOType.D3>
        }
        return main + input
    }

    private val trainChain: (CALC_DELTA_D3) -> CALC_DELTA_D3 by lazy {
        layers.foldRight(
            initial = { final: CALC_DELTA_D3 -> final },
        ) { layer, acc ->
            { final ->
                { input, env ->
                    val scope = this
                    with(layer) {
                        scope._train(input, env) {
                            acc(final)(it as Batch<IOType.D3>, env)
                        }
                    } as Batch<IOType.D3>
                }
            }
        }
    }

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        env: GraphEnv,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        var skipDelta: Batch<IOType.D3>? = null

        val final: CALC_DELTA_D3 = { acc, _ ->
            val output = input + acc
            calcDelta(output).also { skipDelta = it }
        }
        val mainDelta = trainChain(final)(input, env)

        return mainDelta + skipDelta!!
    }

    override fun update(optimizer: Optimizer) {
        layers.forEach { it.update(optimizer) }
    }
}

fun GraphBuilder.Node.D3.skip(
    id: String = Uuid.random().toString(),
    builder: GraphBuilder.Node.D3.() -> GraphBuilder.Node.D3,
): GraphBuilder.Node.D3 {
    val before = nodes.size
    val layers = builder().nodes.drop(before).map { (it as Graph.Node.Attach).process }
    val (outputI, outputJ, outputK) = when (val last = layers.lastOrNull()) {
        is Compute.D3 -> Triple(last.outputI, last.outputJ, last.outputK)
        is Reshape.D1ToD3 -> Triple(last.outputI, last.outputJ, last.outputK)
        is Reshape.D2ToD3 -> Triple(last.outputI, last.outputJ, last.outputK)
        null -> return this
        else -> throw IllegalArgumentException("invalid last layer. $last")
    }

    check(inputI == outputI && inputJ == outputJ && inputK == outputK) {
        """
            invalid parameter.
            input: ($inputI, $inputJ, $inputK)
            output: ($outputI, $outputJ. $outputK)
        """.trimIndent()
    }

    return addCompute(
        compute = SkipD3(
            inputI = outputI,
            inputJ = outputJ,
            inputK = outputK,
            layers = layers,
            id = id,
        ),
    )
}
