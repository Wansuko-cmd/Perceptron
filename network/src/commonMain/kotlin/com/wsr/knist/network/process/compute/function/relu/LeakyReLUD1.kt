package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.GraphBuilder
import com.wsr.knist.network.GraphScope.addCompute
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.process.Compute
import com.wsr.knist.network.process.Context
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class LeakyReLUD1 internal constructor(override val inputI: Int, override val id: String = Uuid.random().toString()) :
    Compute.D1() {
    override val outputI: Int get() = inputI
    override fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> {
        val mask = input gt 0f
        return input.where(condition = mask, onFalse = 0.01f * input)
    }

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        val mask = input gt 0f
        val output = input.where(condition = mask, onFalse = 0.01f * input)
        val delta = calcDelta(output)
        return delta.where(condition = mask, onFalse = 0.01f * delta)
    }
}

fun <T> NetworkBuilder.D1<T>.leakyReLU(id: String = Uuid.random().toString()) = addCompute(
    LeakyReLUD1(inputI = inputI, id = id),
)

fun GraphBuilder.Node.D1.leakyReLU(id: String = Uuid.random().toString()) = addCompute(
    LeakyReLUD1(inputI = inputI, id = id),
)
