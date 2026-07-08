package com.wsr.knist.network.process.compute.conv

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.reshapeToD3
import com.wsr.knist.batch.shape.reshapeToD4
import com.wsr.knist.batch.shape.toBatch
import com.wsr.knist.batch.shape.toD4
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.shape.reshapeToD2
import com.wsr.knist.core.shape.reshapeToD4
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlin.uuid.Uuid
import kotlinx.serialization.Serializable

@Serializable
class ConvD2 internal constructor(
    private val filter: Int,
    private val channel: Int,
    private val kernel: Int,
    private val stride: Int,
    private val padding: Int,
    private val height: Int,
    private val width: Int,
    private val optimizer: Optimizer.D4,
    private var weight: IOType.D4.Global,
    override val id: String = Uuid.random().toString(),
) : Compute.D3() {
    override val inputI: Int = channel
    override val inputJ: Int = height
    override val inputK: Int = width

    override val outputI: Int = filter
    override val outputJ: Int = (height - kernel + 2 * padding) / stride + 1
    override val outputK: Int = (width - kernel + 2 * padding) / stride + 1

    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> {
        val col = input.unfold(windowSize = kernel, stride = stride, padding = padding)
            .reshapeToD3(i = channel, j = outputJ * outputK, k = kernel * kernel)
            .toD4()
            .transpose(axisI = 1, axisJ = 3, axisK = 0, axisL = 2)
            .reshapeToD2(i = channel * kernel * kernel, j = input.size * outputJ * outputK)
        return (weight.reshapeToD2(outputI, channel * kernel * kernel).matMul(col))
            .reshapeToD4(i = filter, j = input.size, k = outputJ, l = outputK)
            .transpose(axisI = 1, axisJ = 0, axisK = 3, axisL = 2)
            .toBatch()
    }

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val col = input.unfold(windowSize = kernel, stride = stride, padding = padding)
            .reshapeToD3(i = channel, j = outputJ * outputK, k = kernel * kernel)
            .toD4()
            .transpose(axisI = 1, axisJ = 3, axisK = 0, axisL = 2)
            .reshapeToD2(i = channel * kernel * kernel, j = input.size * outputJ * outputK)
        val output = (weight.reshapeToD2(i = outputI, j = channel * kernel * kernel).matMul(col))
            .reshapeToD4(i = filter, j = input.size, k = outputJ, l = outputK)
            .transpose(axisI = 1, axisJ = 0, axisK = 3, axisL = 2)
            .toBatch()

        val delta = calcDelta(output)

        val reversed = weight
            .flip(axis = 2)
            .flip(axis = 3)
            .reshapeToD2(i = filter, j = channel * kernel * kernel)
            .transpose()
        val deltaCol = delta.toD4()
            .transpose(axisI = 1, axisJ = 0, axisK = 3, axisL = 2)
            .reshapeToD2(i = filter, j = input.size * outputJ * outputK)
        val dx = (reversed.matMul(deltaCol))
            .reshapeToD4(i = channel, j = kernel * kernel, k = input.size, l = outputJ * outputK)
            .transpose(axisI = 2, axisJ = 0, axisK = 3, axisL = 1)
            .toBatch()
            .reshapeToD4(i = channel, j = outputJ, k = outputK, l = kernel * kernel)
            .fold(stride = stride, padding = padding)

        val dw = deltaCol.matMul(col.transpose())
            .reshapeToD4(i = filter, j = channel, k = kernel, l = kernel)
        weight = optimizer.adapt(weight = weight, dw = dw / input.size.toFloat()).toGlobal()

        return dx
    }

    override fun freeze(isFrozen: Boolean) {
        optimizer.isFrozen = isFrozen
    }
}

fun <T> NetworkBuilder.D3<T>.convD2(
    filter: Int,
    kernel: Int,
    stride: Int = 1,
    padding: Int = 0,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = this.initializer,
    id: String = Uuid.random().toString(),
) = addProcess(
    process =
        ConvD2(
            filter = filter,
            channel = inputI,
            kernel = kernel,
            stride = stride,
            padding = padding,
            height = inputJ,
            width = inputK,
            optimizer = optimizer.d4(filter, inputI, kernel, kernel),
            weight = initializer.d4(
                input = listOf(inputI, kernel, kernel),
                output = listOf(filter, kernel, kernel),
                i = filter,
                j = inputI,
                k = kernel,
                l = kernel,
            ),
            id = id,
        ),
)
