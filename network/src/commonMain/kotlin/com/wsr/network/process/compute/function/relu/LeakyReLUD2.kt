package com.wsr.network.process.compute.function.relu

import com.wsr.batch.Batch
import com.wsr.batch.compare.greater.gt
import com.wsr.batch.compare.where.where
import com.wsr.batch.get
import com.wsr.batch.operation.times.times
import com.wsr.core.IOType
import com.wsr.core.get
import com.wsr.network.NetworkBuilder
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import com.wsr.scope.IOScope
import kotlinx.serialization.Serializable

@Serializable
class LeakyReLUD2 internal constructor(override val outputX: Int, override val outputY: Int) : Compute.D2() {
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val mask = input gt 0f
        return input.where(condition = mask, onFalse = 0.01f)
    }

    override fun IOScope.train(
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
