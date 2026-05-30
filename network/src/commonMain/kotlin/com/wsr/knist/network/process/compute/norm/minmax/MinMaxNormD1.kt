package com.wsr.knist.network.process.compute.norm.minmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.math.pow
import com.wsr.knist.batch.elementwise.operation.div.div
import com.wsr.knist.batch.elementwise.operation.minus.minus
import com.wsr.knist.batch.elementwise.operation.times.times
import com.wsr.knist.batch.get
import com.wsr.knist.batch.linalg.inner
import com.wsr.knist.batch.reduction.max
import com.wsr.knist.batch.reduction.min
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.get
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class MinMaxNormD1 internal constructor(
    override val outputSize: Int,
    private val optimizer: Optimizer.D1,
    private var weight: IOType.D1,
) : Compute.D1() {
    override fun expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> {
        val min = input.min()
        val max = input.max()
        val denominator = max - min
        return weight * (input - min) / denominator
    }

    override fun train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        val min = input.min()
        val max = input.max()

        val numerator = input - min
        val denominator = 1f / (max - min)

        val mean = denominator * numerator
        val output = weight * mean

        val delta = calcDelta(output)

        val dOutput = delta * weight

        weight = optimizer.adapt(
            weight = weight,
            dw = mean * delta,
        )

        numerator.inner(other = dOutput)

        // 分母側(dy/d[max(x) - min(x)])
        val dDenominator = denominator.pow(2) * numerator.inner(other = dOutput)

        // 分子側(dy/d[x - min(x)])
        val dNumerator = denominator * dOutput

        return Batch(input.size) {
            val input = input[it]
            val min = min[it]
            val max = max[it]
            val dDenominator = dDenominator[it]
            val dNumerator = dNumerator[it]
            IOType.d1(input.shape) { x ->
                /**
                 * dy/input + dy/min(x) + dy/max(x)
                 * dy/dx = dNumerator
                 * dy/min(x) = if(x == min(x)) -dNumerator + dDenominator else 0f
                 * dy/max(x) = if(x == max(x)) -dDenominator else 0f
                 */
                val input = input[x]
                when (input) {
                    min.get() -> dDenominator.get()
                    max.get() -> dNumerator[x] - dDenominator.get()
                    else -> dNumerator[x]
                }
            }
        }
    }
}

fun <T> NetworkBuilder.D1<T>.minMaxNorm(
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = Fixed(1f),
) = addProcess(
    process = MinMaxNormD1(
        outputSize = inputSize,
        optimizer = optimizer.d1(inputSize),
        weight = initializer.d1(
            input = listOf(inputSize),
            output = listOf(inputSize),
            size = inputSize,
        ),
    ),
)
