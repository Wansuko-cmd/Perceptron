@file:Suppress("UNCHECKED_CAST")

package com.wsr.network.process.compute.skip

import com.wsr.batch.Batch
import com.wsr.batch.elementwise.operation.plus.plus
import com.wsr.core.IOType
import com.wsr.network.NetworkBuilder
import com.wsr.network.process.Context
import com.wsr.network.process.Process
import com.wsr.network.process.compute.Compute
import kotlinx.serialization.Serializable

private typealias CALC_DELTA_D3 = (input: Batch<IOType.D3>, context: Context) -> Batch<IOType.D3>

@Serializable
class SkipD3 internal constructor(
    // List<Process.D3>だがSerializer対策
    private val layers: List<Process> = emptyList(),
    override val outputX: Int,
    override val outputY: Int,
    override val outputZ: Int,
) : Compute.D3() {
    override fun expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> {
        val main = layers.fold(input) { acc, layer -> layer._expect(acc, context) as Batch<IOType.D3> }
        return main + input
    }

    private val trainChain: (CALC_DELTA_D3) -> CALC_DELTA_D3 by lazy {
        layers.foldRight(
            initial = { final: CALC_DELTA_D3 -> final },
        ) { layer, acc ->
            { final ->
                { input, context ->
                    layer._train(input, context) { acc(final)(it as Batch<IOType.D3>, context) } as Batch<IOType.D3>
                }
            }
        }
    }

    override fun train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        var skipDelta: Batch<IOType.D3>? = null

        val final: CALC_DELTA_D3 = { acc, _ ->
            val output = input + acc
            calcDelta(output).also { skipDelta = it }
        }
        val mainDelta = trainChain(final)(input, context)

        return mainDelta + skipDelta!!
    }
}

fun <T> NetworkBuilder.D3<T>.skip(builder: NetworkBuilder.D3<T>.() -> NetworkBuilder.D3<T>): NetworkBuilder.D3<T> {
    val layers = builder().layers
        .drop(layers.size)
        .filterIsInstance<Compute.D3>()
    val last = layers.last()

    check(inputX == last.outputX && inputY == last.outputY && inputZ == last.outputZ) {
        """
            invalid parameter.
            input: ($inputX, $inputY, $inputZ)
            output: (${last.outputX}, ${last.outputY}. ${last.outputZ})
        """.trimIndent()
    }

    return addProcess(
        process = SkipD3(
            outputX = last.outputX,
            outputY = last.outputY,
            outputZ = last.outputZ,
            layers = layers,
        ),
    )
}
