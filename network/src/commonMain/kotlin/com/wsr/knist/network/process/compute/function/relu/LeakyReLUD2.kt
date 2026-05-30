package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.compare.gt
import com.wsr.knist.batch.elementwise.compare.where.where
import com.wsr.knist.batch.elementwise.operation.times.times
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.get
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class LeakyReLUD2 internal constructor(override val outputX: Int, override val outputY: Int) : Compute.D2() {
    override fun expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val mask = input gt 0f
        return input.where(condition = mask, onFalse = 0.01f)
    }

    override fun train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val mask = input gt 0f
        val output = input.where(condition = mask, onFalse = 0f)
        val delta = calcDelta(output)
        return delta.where(condition = mask, onFalse = 0.01f * delta)
    }
}

fun <T> NetworkBuilder.D2<T>.leakyReLU() = addProcess(LeakyReLUD2(outputX = inputX, outputY = inputY))
