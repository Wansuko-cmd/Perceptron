package com.wsr.network.process.compute.affine

import com.wsr.batch.Batch
import com.wsr.batch.operation.matmul.matMul
import com.wsr.core.IOType
import com.wsr.network.NetworkBuilder
import com.wsr.network.initializer.WeightInitializer
import com.wsr.network.optimizer.Optimizer
import com.wsr.network.process.Context
import com.wsr.network.process.compute.Compute
import com.wsr.scope.IOScope
import kotlinx.serialization.Serializable

@Serializable
class AffineD2 internal constructor(
    private val channel: Int,
    private val outputSize: Int,
    private val optimizer: Optimizer.D2,
    private var weight: IOType.D2,
) : Compute.D2() {
    override val outputX = channel
    override val outputY = outputSize

    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = forward(input)

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val output = forward(input)
        val delta = calcDelta(output)

        val dx = delta.matMul(weight, transB = true)
        val dw = input.matMul(delta, transA = true)

        weight = optimizer.adapt(weight = weight, dw = dw)
        return dx
    }

    private fun forward(input: Batch<IOType.D2>): Batch<IOType.D2> = input.matMul(weight)
}

fun <T> NetworkBuilder.D2<T>.affine(
    neuron: Int,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = this.initializer,
) = addProcess(
    process =
    AffineD2(
        channel = inputX,
        outputSize = neuron,
        optimizer = optimizer.d2(inputY, neuron),
        weight = initializer.d2(
            input = listOf(inputY),
            output = listOf(neuron),
            x = inputY,
            y = neuron,
        ),
    ),
)
