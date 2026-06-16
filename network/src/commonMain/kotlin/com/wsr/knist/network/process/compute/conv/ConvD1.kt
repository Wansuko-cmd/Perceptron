package com.wsr.knist.network.process.compute.conv

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.toBatch
import com.wsr.knist.batch.shape.toD3
import com.wsr.knist.batch.shape.toD4
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.shape.reshapeToD2
import com.wsr.knist.core.shape.reshapeToD3
import com.wsr.knist.core.shape.reshapeToD4
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.Compute
import kotlinx.serialization.Serializable

@Serializable
class ConvD1 internal constructor(
    private val filter: Int,
    private val channel: Int,
    private val kernel: Int,
    private val stride: Int,
    private val padding: Int,
    private val inputSize: Int,
    private val optimizer: Optimizer.D3,
    private var weight: IOType.D3.Global,
) : Compute.D2() {
    override val outputX: Int = filter
    override val outputY: Int = (inputSize - kernel + 2 * padding) / stride + 1

    init {
        check((inputSize - kernel + 2 * padding) % stride == 0) {
            val output = (inputSize - kernel + 2 * padding) / stride.toFloat() + 1.0
            """
            invalid parameter.
            inputSize: $inputSize
            kernel: $kernel
            padding: $padding
            stride: $stride
            output: (inputSize - kernel + 2 * padding) % stride + 1 = $output
            """.trimIndent()
        }
    }

    override fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D2> {
        val col = input.unfold(windowSize = kernel, stride = stride, padding = padding)
            .toD4()
            .transpose(axisI = 1, axisJ = 3, axisK = 0, axisL = 2)
            .reshapeToD2(i = kernel * channel, j = outputY * input.size)
        return (weight.reshapeToD2(outputX, channel * kernel).matMul(col))
            .reshapeToD3(i = filter, j = input.size, k = outputY)
            .transpose(axisI = 1, axisJ = 0, axisK = 2)
            .toBatch()
    }

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        context: Context,
        calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val col = input.unfold(windowSize = kernel, stride = stride, padding = padding)
            .toD4()
            .transpose(axisI = 1, axisJ = 3, axisK = 0, axisL = 2)
            .reshapeToD2(i = kernel * channel, j = outputY * input.size)
        val output = (weight.reshapeToD2(i = outputX, j = channel * kernel).matMul(col))
            .reshapeToD3(i = filter, j = input.size, k = outputY)
            .transpose(axisI = 1, axisJ = 0, axisK = 2)
            .toBatch()

        val delta = calcDelta(output)

        val reversed = weight
            .flip(axis = 2)
            .reshapeToD2(i = filter, j = channel * kernel)
            .transpose()
        val deltaCol = delta.toD3()
            .transpose(axisI = 1, axisJ = 0, axisK = 2)
            .reshapeToD2(i = filter, j = input.size * outputY)
        val dx = (reversed.matMul(deltaCol))
            .reshapeToD4(i = channel, j = kernel, k = input.size, l = outputY)
            .transpose(axisI = 2, axisJ = 0, axisK = 3, axisL = 1)
            .toBatch()
            .fold(stride = stride, padding = padding)

        val dw = deltaCol.matMul(col.transpose())
            .reshapeToD3(i = filter, j = channel, k = kernel)
        weight = optimizer.adapt(weight = weight, dw = dw / input.size.toFloat()).toGlobal()

        return dx
    }
}

fun <T> NetworkBuilder.D2<T>.convD1(
    filter: Int,
    kernel: Int,
    stride: Int = 1,
    padding: Int = 0,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = this.initializer,
) = addProcess(
    process =
        ConvD1(
            filter = filter,
            channel = inputX,
            kernel = kernel,
            stride = stride,
            padding = padding,
            inputSize = inputY,
            optimizer = optimizer.d3(filter, inputX, kernel),
            weight = initializer.d3(
                input = listOf(inputX, kernel),
                output = listOf(filter, kernel),
                x = filter,
                y = inputX,
                z = kernel,
            ),
        ),
)
