package com.wsr.network.process.compute.scale.d3

import com.wsr.batch.Batch
import com.wsr.batch.collecction.sum.sum
import com.wsr.batch.operation.times.times
import com.wsr.core.IOType
import com.wsr.network.optimizer.Optimizer
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import com.wsr.scope.IOScope
import kotlinx.serialization.Serializable

@Serializable
class ScaleAxisD3 internal constructor(
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
    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> =
        input.times(other = weight, axis = axis)

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val output = input.times(other = weight, axis = axis)
        val delta = calcDelta(output)

        val dx = delta.times(other = weight, axis = axis)
        weight = optimizer.adapt(
            weight = weight,
            dw = (input * delta).sum(axis = sumAxis1).sum(axis = sumAxis2),
        )

        return dx
    }
}
