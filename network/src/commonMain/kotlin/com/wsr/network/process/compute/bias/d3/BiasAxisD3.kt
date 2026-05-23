package com.wsr.network.process.compute.bias.d3

import com.wsr.batch.Batch
import com.wsr.batch.reduction.sum
import com.wsr.batch.elementwise.operation.plus.plus
import com.wsr.core.IOType
import com.wsr.network.optimizer.Optimizer
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class BiasAxisD3(
    override val outputX: Int,
    override val outputY: Int,
    override val outputZ: Int,
    private val axis: Int,
    private val optimizer: Optimizer.D1,
    private var weight: IOType.D1,
) : Compute.D3() {
    private val sumAxis1 = when (axis) {
        0 -> 1
        else -> 0
    }
    private val sumAxis2 = when (axis) {
        0, 1 -> 1
        else -> 0
    }

    override fun expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> =
        input.plus(other = weight, axis = axis)

    override fun train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val output = input.plus(other = weight, axis = axis)
        val delta = calcDelta(output)
        weight = optimizer.adapt(weight = weight, dw = delta.sum(sumAxis1).sum(sumAxis2))
        return delta
    }
}
