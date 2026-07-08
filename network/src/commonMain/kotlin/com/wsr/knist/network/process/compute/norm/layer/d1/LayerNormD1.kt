package com.wsr.knist.network.process.compute.norm.layer.d1

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class LayerNormD1 internal constructor(
    override val inputI: Int,
    private val e: Float,
    override val id: String = Uuid.random().toString(),
) : Compute.D1() {
    override val outputI: Int get() = inputI
    override fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> {
        val average = input.average()
        val numerator = input - average

        val variance = numerator.pow(n = 2).average()
        val denominator = variance.sqrt(e = e)

        return numerator / denominator
    }

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        val average = input.average()
        val numerator = input - average

        val variance = numerator.pow(n = 2).average()
        val denominator = variance.sqrt(e = e)

        val output = numerator / denominator
        val delta = calcDelta(output)

        // dy/[x-average(x)]
        val dNumerator = delta / denominator

        // dy/x <- (x-average(x)のx)
        val dx1 = dNumerator

        // dy/x <- average(x)のx
        val dx2 = -1f * dNumerator.sum() / outputI.toFloat()

        // dy/x <- variance(x)のx
        val dx3 = run {
            /**
             * dy/[sqrt(variance(x)]
             *   = (sum(delta * numerator) / denominator) * (-1 / (2f * denominator^2))
             *   = -sum(delta * numerator / denominator) / 2f * denominator^2
             *   = -sum(delta * output) / denominator^2
             *
             * d[sqrt(variance(x)]/[variance(x)] = 1 / outputSize
             *
             * dy/[variance(x)]
             *   = -sum(delta * output) / (denominator^2 * outputSize)
             */
            val dvn = -1f * (delta * output).sum()
            val dvd = 2f * denominator.pow(2) * outputI.toFloat()
            val dVariance = dvn / dvd

            // dy/[x-average(x)]
            val dSquared = 2f * dVariance * numerator

            // dy/[x]
            val dx1 = dSquared
            // dy/[-average(x)]
            val dx2 = -1f * dSquared.sum() / outputI.toFloat()

            dx1 + dx2
        }

        // dy/dx
        return dx1 + dx2 + dx3
    }
}

fun <T> NetworkBuilder.D1<T>.layerNorm(e: Float = 1e-6f, id: String = Uuid.random().toString()) = addProcess(
    process = LayerNormD1(inputI = inputI, e = e, id = id),
)
