package com.wsr.network.process.compute.norm.rms.d1

import com.wsr.batch.Batch
import com.wsr.batch.reduction.average.average
import com.wsr.batch.elementwise.math.pow
import com.wsr.batch.elementwise.math.sqrt
import com.wsr.batch.elementwise.operation.div.div
import com.wsr.batch.elementwise.operation.minus.minus
import com.wsr.batch.elementwise.operation.times.times
import com.wsr.core.IOType
import com.wsr.network.NetworkBuilder
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class RmsNormD1 internal constructor(override val outputSize: Int, private val e: Float) : Compute.D1() {
    override fun expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> {
        val deviation = input.pow(n = 2).average().sqrt(e = e)
        return input / deviation
    }

    override fun train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
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

fun <T> NetworkBuilder.D1<T>.rmsNorm(e: Float = 1e-6f) = addProcess(process = RmsNormD1(outputSize = inputSize, e = e))
