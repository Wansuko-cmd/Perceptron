package com.wsr.network.process.compute.dropout

import com.wsr.batch.Batch
import com.wsr.batch.operation.times.times
import com.wsr.core.IOType
import com.wsr.core.d3
import com.wsr.network.NetworkBuilder
import com.wsr.network.nextFloat
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import com.wsr.scope.IOScope
import kotlin.random.Random
import kotlinx.serialization.Serializable

@Serializable
class DropoutD3 internal constructor(
    override val outputX: Int,
    override val outputY: Int,
    override val outputZ: Int,
    private val ratio: Float,
    private val seed: Int? = null,
) : Compute.D3() {
    private val random by lazy { seed?.let { Random(it) } ?: Random }
    private val q = 1 / ratio

    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> = input

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val mask = IOType.d3(
            i = outputX,
            j = outputY,
            k = outputZ,
        ) { _, _, _ -> if (random.nextFloat(0f, 1f) <= ratio) q else 0f }
        val output = input * mask
        val delta = calcDelta(output)
        return delta * mask
    }
}

fun <T> NetworkBuilder.D3<T>.dropout(ratio: Float, seed: Int? = null) = addProcess(
    process =
    DropoutD3(
        outputX = inputX,
        outputY = inputY,
        outputZ = inputZ,
        ratio = ratio,
        seed = seed,
    ),
)
