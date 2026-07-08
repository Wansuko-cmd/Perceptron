package com.wsr.knist.network.process.compute.bias.d2

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class BiasAxisD2(
    override val outputI: Int,
    override val outputJ: Int,
    private val axis: Int,
    private val optimizer: Optimizer.D1,
    private var weight: IOType.D1.Global,
    override val id: String = Uuid.random().toString(),
) : Compute.D2() {
    private val sumAxis = if (axis == 0) 1 else 0
    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> =
        input.plus(other = weight, axis = axis)

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val output = input.plus(other = weight, axis = axis)
        val delta = calcDelta(output)
        weight = optimizer.adapt(weight = weight, dw = delta.sum(axis = sumAxis)).toGlobal()
        return delta
    }
}
