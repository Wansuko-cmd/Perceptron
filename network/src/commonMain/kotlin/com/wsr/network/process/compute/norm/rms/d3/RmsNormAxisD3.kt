package com.wsr.network.process.compute.norm.rms.d3

import com.wsr.batch.Batch
import com.wsr.batch.elementwise.math.pow
import com.wsr.batch.elementwise.math.sqrt
import com.wsr.batch.elementwise.operation.div.div
import com.wsr.batch.elementwise.operation.minus.minus
import com.wsr.batch.elementwise.operation.times.times
import com.wsr.batch.reduction.average.average
import com.wsr.core.IOType
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class RmsNormAxisD3 internal constructor(
    override val outputX: Int,
    override val outputY: Int,
    override val outputZ: Int,
    private val axis: Int,
    private val e: Float,
) : Compute.D3() {
    private val axis1 = when (axis) {
        0 -> 1
        else -> 0
    }
    private val axis2 = when (axis) {
        0, 1 -> 2
        else -> 1
    }

    override fun expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> {
        val deviation = input.pow(n = 2).average(axis = axis).sqrt(e = e)
        return input.div(other = deviation, axis1 = axis1, axis2 = axis2)
    }

    override fun train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val variance = input.pow(2).average(axis = axis)
        val deviation = variance.sqrt(e = e)
        val output = input.div(other = deviation, axis1 = axis1, axis2 = axis2)

        val delta = calcDelta(output)

        val dx1 = delta.div(other = deviation, axis1 = axis1, axis2 = axis2)
        val dx2 = run {
            val m = (delta * input).average(axis = axis) / deviation.pow(3)
            m.times(other = input, axis1 = axis1, axis2 = axis2)
        }
        return dx1 - dx2
    }
}
