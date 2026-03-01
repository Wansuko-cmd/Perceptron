@file:Suppress("UNCHECKED_CAST")

package com.wsr.network.process.compute.skip

import com.wsr.batch.Batch
import com.wsr.batch.operation.plus.plus
import com.wsr.core.IOType
import com.wsr.network.NetworkBuilder
import com.wsr.network.process.Context
import com.wsr.network.process.Process
import com.wsr.network.process.compute.Compute
import kotlinx.serialization.Serializable

private typealias CALC_DELTA_D1 = (input: Batch<IOType.D1>, context: Context) -> Batch<IOType.D1>

@Serializable
class SkipD1 internal constructor(
    // List<Process.D1>だがSerializer対策
    private val layers: List<Process> = emptyList(),
    override val outputSize: Int,
) : Compute.D1() {
    override fun expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> {
        val main = layers.fold(input) { acc, layer -> layer._expect(acc, context) as Batch<IOType.D1> }
        return main + input
    }

    private val trainChain: (CALC_DELTA_D1) -> CALC_DELTA_D1 by lazy {
        layers.foldRight(
            initial = { final: CALC_DELTA_D1 -> final },
        ) { layer, acc ->
            { final ->
                { input, context ->
                    layer._train(input, context) { acc(final)(it as Batch<IOType.D1>, context) } as Batch<IOType.D1>
                }
            }
        }
    }

    override fun train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        var skipDelta: Batch<IOType.D1>? = null

        val final: CALC_DELTA_D1 = { acc, _ ->
            val output = input + acc
            calcDelta(output).also { skipDelta = it }
        }
        val mainDelta = trainChain(final)(input, context)

        return mainDelta + skipDelta!!
    }
}

fun <T> NetworkBuilder.D1<T>.skip(builder: NetworkBuilder.D1<T>.() -> NetworkBuilder.D1<T>): NetworkBuilder.D1<T> {
    val layers = builder().layers
        .drop(layers.size)
        .filterIsInstance<Compute.D1>()
    val last = layers.last()

    check(inputSize == last.outputSize) {
        """
            invalid parameter.
            input: ($inputSize)
            output: (${last.outputSize})
        """.trimIndent()
    }

    return addProcess(
        process = SkipD1(
            outputSize = last.outputSize,
            layers = layers,
        ),
    )
}
