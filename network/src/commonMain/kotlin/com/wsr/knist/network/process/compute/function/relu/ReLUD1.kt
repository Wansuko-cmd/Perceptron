package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class ReLUD1 internal constructor(override val outputI: Int, override val id: String = Uuid.random().toString()) :
    Compute.D1() {
    override fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> {
        val mask = input gt 0f
        return input.where(condition = mask, onFalse = 0f)
    }

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        val mask = input gt 0f
        val output = input.where(condition = mask, onFalse = 0f)
        val delta = calcDelta(output)
        return delta.where(condition = mask, onFalse = 0f)
    }
}

fun <T> NetworkBuilder.D1<T>.reLU(id: String = Uuid.random().toString()) = addProcess(ReLUD1(outputI = inputI, id = id))
