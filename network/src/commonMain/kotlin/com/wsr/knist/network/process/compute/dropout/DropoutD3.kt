package com.wsr.knist.network.process.compute.dropout

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.nextFloat
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.random.Random
import kotlinx.serialization.Serializable

@Serializable
class DropoutD3 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    override val outputK: Int,
    private val ratio: Float,
    private val seed: Int? = null,
) : Compute.D3() {
    private val random by lazy { seed?.let { Random(it) } ?: Random }
    private val q = 1 / ratio

    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = input

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val mask = Batch.d3(
            size = input.size,
            i = outputI,
            j = outputJ,
            k = outputK,
        ) { _, _, _ -> if (random.nextFloat(0f, 1f) <= ratio) q else 0f }
        val output = input * mask
        val delta = calcDelta(output)
        return delta * mask
    }
}

fun <T> NetworkBuilder.D3<T>.dropout(ratio: Float, seed: Int? = null) = addProcess(
    process =
        DropoutD3(
            outputI = inputI,
            outputJ = inputJ,
            outputK = inputK,
            ratio = ratio,
            seed = seed,
        ),
)
