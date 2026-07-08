@file:Suppress("UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.skip

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.Process
import com.wsr.knist.network.process.compute.Compute
import com.wsr.knist.network.process.reshape.Reshape
import com.wsr.knist.network.process.reshape.reshape.ReshapeD2ToD1
import com.wsr.knist.network.process.reshape.reshape.ReshapeD3ToD1
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

private typealias CALC_DELTA_D1 = IOScope.(input: Batch<IOType.D1>, context: Context) -> Batch<IOType.D1>

@Serializable
class SkipD1 internal constructor(
    // List<Process.D1>だがSerializer対策
    private val layers: List<Process> = emptyList(),
    override val inputI: Int,
    override val id: String = Uuid.random().toString(),
) : Compute.D1() {
    override val outputI: Int get() = inputI
    override fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> {
        val scope = this
        val main = layers.fold(input) { acc, layer ->
            with(layer) { scope._expect(acc, context) } as Batch<IOType.D1>
        }
        return main + input
    }

    private val trainChain: (CALC_DELTA_D1) -> CALC_DELTA_D1 by lazy {
        layers.foldRight(
            initial = { final: CALC_DELTA_D1 -> final },
        ) { layer, acc ->
            { final ->
                { input, context ->
                    val scope = this
                    with(layer) {
                        scope._train(input, context) {
                            acc(final)(it as Batch<IOType.D1>, context)
                        }
                    } as Batch<IOType.D1>
                }
            }
        }
    }

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        var skipDelta: Batch<IOType.D1>? = null

        val final: CALC_DELTA_D1 = { acc, _ ->
            val output = input + acc
            calcDelta(output).also { skipDelta = it }
        }
        val mainDelta = trainChain(final)(input, context)

        return mainDelta + skipDelta!!
    }

    override fun freeze(isFrozen: Boolean) {
        layers.forEach { it.freeze(isFrozen) }
    }
}

fun <T> NetworkBuilder.D1<T>.skip(
    id: String = Uuid.random().toString(),
    builder: NetworkBuilder.D1<T>.() -> NetworkBuilder.D1<T>,
): NetworkBuilder.D1<T> {
    val layers = builder().layers.drop(layers.size)
    val outputI = when (val last = layers.lastOrNull()) {
        is Compute.D1 -> last.outputI
        is Reshape.D2ToD1 -> last.outputI
        is Reshape.D3ToD1 -> last.outputI
        null -> return this
        else -> throw IllegalArgumentException("invalid last layer. $last")
    }

    check(inputI == outputI) {
        """
            invalid parameter.
            input: ($inputI)
            output: ($outputI)
        """.trimIndent()
    }

    return addProcess(
        process = SkipD1(
            inputI = outputI,
            layers = layers,
            id = id,
        ),
    )
}
