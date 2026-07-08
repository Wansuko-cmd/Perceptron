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
class MinMaxNormD2 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    private val optimizer: Optimizer.D2,
    private var weight: IOType.D2.Global,
    override val id: String = Uuid.random().toString(),
) : Compute.D2() {
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val min = input.min()
        val max = input.max()
        val denominator = max - min
        return weight * (input - min) / denominator
    }

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
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

        // 分母側(dy/d[max(x) - min(x)])
        val dDenominator = denominator.pow(2) * (numerator * dOutput).sum()

        // 分子側(dy/d[x - min(x)])
        val dNumerator = denominator * dOutput

        val dMin = dDenominator.broadcastToD2(input.shape)
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

fun <T> NetworkBuilder.D2<T>.minMaxNorm(
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = Fixed(1f),
    id: String = Uuid.random().toString(),
) = addProcess(
    process =
        MinMaxNormD2(
            outputI = inputI,
            outputJ = inputJ,
            optimizer = optimizer.d2(inputI, inputJ),
            weight = initializer.d2(
                input = listOf(inputI, inputJ),
                output = listOf(inputI, inputJ),
                i = inputI,
                j = inputJ,
            ),
            id = id,
        ),
)
