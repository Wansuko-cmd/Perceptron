@file:Suppress("UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.skip

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Compute
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.Process
import com.wsr.knist.network.process.Reshape
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

private typealias CALC_DELTA_D2 = IOScope.(input: Batch<IOType.D2>, context: Context) -> Batch<IOType.D2>

@Serializable
class SkipD2 internal constructor(
    // List<Process.D2>だがSerializer対策
    private val layers: List<Process> = emptyList(),
    override val inputI: Int,
    override val inputJ: Int,
    override val id: String = Uuid.random().toString(),
) : Compute.D2() {
    override val outputI: Int get() = inputI
    override val outputJ: Int get() = inputJ
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val scope = this
        val main = layers.fold(input) { acc, layer ->
            with(layer) { scope._expect(acc, context) } as Batch<IOType.D2>
        }
        return main + input
    }

    private val trainChain: (CALC_DELTA_D2) -> CALC_DELTA_D2 by lazy {
        layers.foldRight(
            initial = { final: CALC_DELTA_D2 -> final },
        ) { layer, acc ->
            { final ->
                { input, context ->
                    val scope = this
                    with(layer) {
                        scope._train(input, context) {
                            acc(final)(it as Batch<IOType.D2>, context)
                        }
                    } as Batch<IOType.D2>
                }
            }
        }
    }

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        var skipDelta: Batch<IOType.D2>? = null

        val final: CALC_DELTA_D2 = { acc, _ ->
            val output = input + acc
            calcDelta(output).also { skipDelta = it }
        }
        val mainDelta = trainChain(final)(input, context)

        return mainDelta + skipDelta!!
    }

    override fun freeze(isFrozen: Boolean) {
        layers.forEach { it.freeze(isFrozen) }
    }

    override fun update(optimizer: Optimizer) {
        layers.forEach { it.update(optimizer) }
    }
}

fun <T> NetworkBuilder.D2<T>.skip(
    id: String = Uuid.random().toString(),
    builder: NetworkBuilder.D2<T>.() -> NetworkBuilder.D2<T>,
): NetworkBuilder.D2<T> {
    val layers = builder().layers.drop(layers.size)
    val (outputI, outputJ) = when (val last = layers.lastOrNull()) {
        is Compute.D2 -> last.outputI to last.outputJ
        is Reshape.D1ToD2 -> last.outputI to last.outputJ
        is Reshape.D3ToD2 -> last.outputI to last.outputJ
        null -> return this
        else -> throw IllegalArgumentException("invalid last layer. $last")
    }

    check(inputI == outputI && inputJ == outputJ) {
        """
            invalid parameter.
            input: ($inputI, $inputJ)
            output: ($outputI, $outputJ)
        """.trimIndent()
    }

    return addCompute(
        compute = SkipD2(
            inputI = outputI,
            inputJ = outputJ,
            layers = layers,
            id = id,
        ),
    )
}
