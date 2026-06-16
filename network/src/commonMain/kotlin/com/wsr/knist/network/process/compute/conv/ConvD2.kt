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
import kotlinx.serialization.Serializable

@Serializable
class ConvD2 internal constructor(
    private val filter: Int,
    private val channel: Int,
    private val kernel: Int,
    private val stride: Int,
    private val padding: Int,
    private val inputX: Int,
    private val inputY: Int,
    private val optimizer: Optimizer.D4,
    private var weight: IOType.D4,
) : Compute.D3() {
    override val outputX: Int = filter
    override val outputY: Int = (inputX - kernel + 2 * padding) / stride + 1
    override val outputZ: Int = (inputY - kernel + 2 * padding) / stride + 1
    override fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D3> {
        val col = input.unfold(windowSize = kernel, stride = stride, padding = padding)
            .reshapeToD3(i = channel, j = outputY * outputZ, k = kernel * kernel)
            .toD4()
            .transpose(axisI = 1, axisJ = 3, axisK = 0, axisL = 2)
            .reshapeToD2(i = channel * kernel * kernel, j = input.size * outputY * outputZ)
        return (weight.reshapeToD2(outputX, channel * kernel * kernel).matMul(col))
            .reshapeToD4(i = filter, j = input.size, k = outputY, l = outputZ)
            .transpose(axisI = 1, axisJ = 0, axisK = 3, axisL = 2)
            .toBatch()
    }

    override fun IOScope.train(
        input: Batch<IOType.D3>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val col = input.unfold(windowSize = kernel, stride = stride, padding = padding)
            .reshapeToD3(i = channel, j = outputY * outputZ, k = kernel * kernel)
            .toD4()
            .transpose(axisI = 1, axisJ = 3, axisK = 0, axisL = 2)
            .reshapeToD2(i = channel * kernel * kernel, j = input.size * outputY * outputZ)
        val output = (weight.reshapeToD2(i = outputX, j = channel * kernel * kernel).matMul(col))
            .reshapeToD4(i = filter, j = input.size, k = outputY, l = outputZ)
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
            .reshapeToD2(i = filter, j = input.size * outputY * outputZ)
        val dx = (reversed.matMul(deltaCol))
            .reshapeToD4(i = channel, j = kernel * kernel, k = input.size, l = outputY * outputZ)
            .transpose(axisI = 2, axisJ = 0, axisK = 3, axisL = 1)
            .toBatch()
            .reshapeToD4(i = channel, j = outputY, k = outputZ, l = kernel * kernel)
            .fold(stride = stride, padding = padding)

        val dw = deltaCol.matMul(col.transpose())
            .reshapeToD4(i = filter, j = channel, k = kernel, l = kernel)
        weight = optimizer.adapt(weight = weight, dw = dw / input.size.toFloat())

        return dx
    }
}

fun <T> NetworkBuilder.D3<T>.convD2(
    filter: Int,
    kernel: Int,
    stride: Int = 1,
    padding: Int = 0,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = this.initializer,
) = addProcess(
    process =
        ConvD2(
            filter = filter,
            channel = inputX,
            kernel = kernel,
            stride = stride,
            padding = padding,
            inputX = inputY,
            inputY = inputZ,
            optimizer = optimizer.d4(filter, inputX, kernel, kernel),
            weight = initializer.d4(
                input = listOf(inputX, kernel, kernel),
                output = listOf(filter, kernel, kernel),
                i = filter,
                j = inputX,
                k = kernel,
                l = kernel,
            ),
        ),
)
