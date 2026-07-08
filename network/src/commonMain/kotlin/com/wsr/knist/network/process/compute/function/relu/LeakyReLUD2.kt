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
class LeakyReLUD2 internal constructor(
    override val outputI: Int,
    override val outputJ: Int,
    override val id: String = Uuid.random().toString(),
) : Compute.D2() {
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val mask = input gt 0f
        return input.where(condition = mask, onFalse = 0.01f)
    }

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val mask = input gt 0f
        val output = input.where(condition = mask, onFalse = 0f)
        val delta = calcDelta(output)
        return delta.where(condition = mask, onFalse = 0.01f * delta)
    }
}

fun <T> NetworkBuilder.D2<T>.leakyReLU(id: String = Uuid.random().toString()) = addProcess(
    LeakyReLUD2(outputI = inputI, outputJ = inputJ, id = id),
)
