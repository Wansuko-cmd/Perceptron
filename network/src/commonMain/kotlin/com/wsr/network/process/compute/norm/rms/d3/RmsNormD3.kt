package com.wsr.network.process.compute.norm.rms.d3

import com.wsr.batch.Batch
import com.wsr.batch.collecction.average.average
import com.wsr.batch.math.pow
import com.wsr.batch.math.sqrt
import com.wsr.batch.operation.div.div
import com.wsr.batch.operation.minus.minus
import com.wsr.batch.operation.times.times
import com.wsr.core.IOType
import com.wsr.network.NetworkBuilder
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import com.wsr.scope.IOScope
import kotlinx.serialization.Serializable

@Serializable
class RmsNormD3 internal constructor(
    override val outputX: Int,
    override val outputY: Int,
    override val outputZ: Int,
    private val e: Float,
) : Compute.D3() {
    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> {
        val deviation = input.pow(n = 2).average().sqrt(e = e)
        return input / deviation
    }

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val variance = input.pow(2).average()
        val deviation = variance.sqrt(e = e)
        val output = input / deviation

        val delta = calcDelta(output)

        val dx1 = delta / deviation
        val dx2 = run {
            val m = (delta * input).average() / deviation.pow(3)
            m * input
        }
        return dx1 - dx2
    }
}

fun <T> NetworkBuilder.D3<T>.rmsNorm(axis: Int? = null, e: Float = 1e-6f): NetworkBuilder.D3<T> {
    val process = when (axis) {
        null -> RmsNormD3(
            outputX = inputX,
            outputY = inputY,
            outputZ = inputZ,
            e = e,
        )

        0, 1 -> RmsNormAxisD3(
            outputX = inputX,
            outputY = inputY,
            outputZ = inputZ,
            axis = axis,
            e = e,
        )

        else -> throw IllegalStateException(
            """
            invalid parameter.
            axis: $axis
            """.trimIndent(),
        )
    }
    return addProcess(process = process)
}
