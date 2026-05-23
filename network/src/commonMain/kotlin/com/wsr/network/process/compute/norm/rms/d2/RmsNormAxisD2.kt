package com.wsr.network.process.compute.norm.rms.d2

import com.wsr.batch.Batch
import com.wsr.batch.reduction.average.average
import com.wsr.batch.elementwise.math.pow
import com.wsr.batch.elementwise.math.sqrt
import com.wsr.batch.elementwise.operation.div.div
import com.wsr.batch.elementwise.operation.minus.minus
import com.wsr.batch.elementwise.operation.times.times
import com.wsr.core.IOType
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class RmsNormAxisD2 internal constructor(
    override val outputX: Int,
    override val outputY: Int,
    private val axis: Int,
    private val e: Float,
) : Compute.D2() {
    // 四則演算用
    private val basicOpAxis = if (axis == 0) 1 else 0

    override fun expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val deviation = input.pow(n = 2).average(axis = axis).sqrt(e = e)
        return input.div(other = deviation, axis = basicOpAxis)
    }

    override fun train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val variance = input.pow(2).average(axis = axis)
        val deviation = variance.sqrt(e = e)
        val output = input.div(other = deviation, axis = basicOpAxis)

        val delta = calcDelta(output)

        val dx1 = delta.div(other = deviation, axis = basicOpAxis)
        val dx2 = run {
            val m = (delta * input).average(axis = axis) / deviation.pow(3)
            m.times(other = input, axis = basicOpAxis)
        }
        return dx1 - dx2
    }
}
