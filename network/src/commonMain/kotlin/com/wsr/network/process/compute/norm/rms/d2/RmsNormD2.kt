package com.wsr.network.process.compute.norm.rms.d2

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
class RmsNormD2 internal constructor(override val outputX: Int, override val outputY: Int, private val e: Float) :
    Compute.D2() {
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val deviation = input.pow(n = 2).average().sqrt(e = e)
        return input / deviation
    }

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
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

fun <T> NetworkBuilder.D2<T>.rmsNorm(axis: Int? = null, e: Float = 1e-6f): NetworkBuilder.D2<T> {
    val process = when (axis) {
        null -> RmsNormD2(
            outputX = inputX,
            outputY = inputY,
            e = e,
        )

        0, 1 -> RmsNormAxisD2(
            outputX = inputX,
            outputY = inputY,
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
