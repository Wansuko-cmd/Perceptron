package com.wsr.knist.network.process.compute.affine

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
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

    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> = input.matMul(weight)

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val output = input.matMul(weight)
        val delta = calcDelta(output)

        val dx = delta.matMul(weight, transB = true)
        val dw = input.matMul(delta, transA = true)

        weight = optimizer.adapt(weight = weight, dw = dw)
        return dx
    }
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
