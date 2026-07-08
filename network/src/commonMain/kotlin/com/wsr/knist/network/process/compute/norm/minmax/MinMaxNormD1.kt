package com.wsr.knist.network.process.compute.norm.minmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class MinMaxNormD1 internal constructor(
    override val outputI: Int,
    private val optimizer: Optimizer.D1,
    private var weight: IOType.D1.Global,
    override val id: String = Uuid.random().toString(),
) : Compute.D1() {
    override fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> {
        val min = input.min()
        val max = input.max()
        val denominator = max - min
        return weight * (input - min) / denominator
    }

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
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
        ).toGlobal()

        numerator.inner(other = dOutput)

        // 分母側(dy/d[max(x) - min(x)])
        val dDenominator = denominator.pow(2) * numerator.inner(other = dOutput)

        // 分子側(dy/d[x - min(x)])
        val dNumerator = denominator * dOutput

        val dMin = dDenominator.broadcastToD1(input.step)
        val dMax = dNumerator - dDenominator
        return where(
            condition = (input - min) eq 0f,
            onTrue = dMin,
            onFalse = where(
                condition = (input - max) eq 0f,
                onTrue = dMax,
                onFalse = dNumerator,
            ),
        )
    }
}

fun <T> NetworkBuilder.D1<T>.minMaxNorm(
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = Fixed(1f),
    id: String = Uuid.random().toString(),
) = addProcess(
    process = MinMaxNormD1(
        outputI = inputI,
        optimizer = optimizer.d1(inputI),
        weight = initializer.d1(
            input = listOf(inputI),
            output = listOf(inputI),
            size = inputI,
        ),
        id = id,
    ),
)
