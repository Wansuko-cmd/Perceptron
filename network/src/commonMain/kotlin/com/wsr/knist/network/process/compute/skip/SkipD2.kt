@file:Suppress("UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.skip

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.operation.plus.plus
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.Process
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

private typealias CALC_DELTA_D2 = (input: Batch<IOType.D2>, context: Context) -> Batch<IOType.D2>

@Serializable
class SkipD2 internal constructor(
    // List<Process.D2>だがSerializer対策
    private val layers: List<Process> = emptyList(),
    override val outputX: Int,
    override val outputY: Int,
) : Compute.D2() {
    override fun expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val main = layers.fold(input) { acc, layer -> layer._expect(acc, context) as Batch<IOType.D2> }
        return main + input
    }

    private val trainChain: (CALC_DELTA_D2) -> CALC_DELTA_D2 by lazy {
        layers.foldRight(
            initial = { final: CALC_DELTA_D2 -> final },
        ) { layer, acc ->
            { final ->
                { input, context ->
                    layer._train(input, context) { acc(final)(it as Batch<IOType.D2>, context) } as Batch<IOType.D2>
                }
            }
        }
    }

    override fun train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        var skipDelta: Batch<IOType.D2>? = null

        val final: CALC_DELTA_D2 = { acc, _ ->
            val output = input + acc
            calcDelta(output).also { skipDelta = it }
        }
        val mainDelta = trainChain(final)(input, context)

        return mainDelta + skipDelta!!
    }
}

fun <T> NetworkBuilder.D2<T>.skip(builder: NetworkBuilder.D2<T>.() -> NetworkBuilder.D2<T>): NetworkBuilder.D2<T> {
    val layers = builder().layers
        .drop(layers.size)
        .filterIsInstance<Compute.D2>()
    val last = layers.last()

    check(inputX == last.outputX && inputY == last.outputY) {
        """
            invalid parameter.
            input: ($inputX, $inputY)
            output: (${last.outputX}, ${last.outputY})
        """.trimIndent()
    }

    return addProcess(
        process = SkipD2(
            outputX = last.outputX,
            outputY = last.outputY,
            layers = layers,
        ),
    )
}
