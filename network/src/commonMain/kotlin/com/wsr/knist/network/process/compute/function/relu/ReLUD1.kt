package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.compare.gt
import com.wsr.knist.batch.elementwise.compare.where.where
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.get
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class ReLUD1 internal constructor(override val outputSize: Int) : Compute.D1() {
    override fun expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> {
        val mask = input gt 0f
        return input.where(condition = mask, onFalse = 0f)
    }

    override fun train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        val mask = input gt 0f
        val output = input.where(condition = mask, onFalse = 0f)
        val delta = calcDelta(output)
        return delta.where(condition = mask, onFalse = 0f)
    }
}

fun <T> NetworkBuilder.D1<T>.reLU() = addProcess(ReLUD1(outputSize = inputSize))
