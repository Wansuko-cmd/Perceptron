package com.wsr.knist.network.process.compute.norm.minmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class MinMaxNormD2 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    override val id: String = Uuid.random().toString(),
) : Compute.D2() {
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val min = input.min()
        val max = input.max()
        val denominator = max - min
        return (input - min) / denominator
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

        val output = denominator * numerator
        val delta = calcDelta(output)

        // 分母側(dy/d[max(x) - min(x)])
        val dDenominator = denominator.pow(2) * (numerator * delta).sum()

        // 分子側(dy/d[x - min(x)])
        val dNumerator = denominator * delta

        val dMin = dNumerator + dDenominator - denominator * delta.sum()
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

fun <T> NetworkBuilder.D2<T>.minMaxNorm(id: String = Uuid.random().toString()) = addProcess(
    process =
        MinMaxNormD2(
            outputI = inputI,
            outputJ = inputJ,
            id = id,
        ),
)
