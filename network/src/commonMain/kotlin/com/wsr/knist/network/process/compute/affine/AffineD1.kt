package com.wsr.knist.network.process.compute.affine

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.toD2
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class AffineD1 internal constructor(
    override val outputSize: Int,
    private val optimizer: Optimizer.D2,
    private var weight: IOType.D2,
) : Compute.D1() {
    override fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D1> =
        weight.matMul(input, trans = true)

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        val output = weight.matMul(input, trans = true)
        val delta = calcDelta(output)
        val dx = weight.matMul(delta)
        val dw = input.toD2().matMul(delta.toD2(), transA = true)
        weight = optimizer.adapt(weight = weight, dw = dw)
        return dx
    }
}

fun <T> NetworkBuilder.D1<T>.affine(
    neuron: Int,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = this.initializer,
) = addProcess(
    process =
        AffineD1(
            outputSize = neuron,
            optimizer = optimizer.d2(inputSize, neuron),
            weight = initializer.d2(
                input = listOf(inputSize),
                output = listOf(neuron),
                x = inputSize,
                y = neuron,
            ),
        ),
)
