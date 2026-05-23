package com.wsr.network.output.softmax

import com.wsr.batch.Batch
import com.wsr.batch.elementwise.compare.eq
import com.wsr.batch.elementwise.compare.where.where
import com.wsr.batch.elementwise.math.exp
import com.wsr.batch.elementwise.math.ln
import com.wsr.batch.elementwise.operation.div.div
import com.wsr.batch.elementwise.operation.minus.minus
import com.wsr.batch.elementwise.operation.times.times
import com.wsr.batch.reduction.average.batchAverage
import com.wsr.batch.reduction.max
import com.wsr.batch.reduction.sum
import com.wsr.core.IOType
import com.wsr.core.get
import com.wsr.network.NetworkBuilder
import com.wsr.network.converter.Converter
import com.wsr.network.output.Output
import com.wsr.network.output.TResult
import kotlinx.serialization.Serializable

@Serializable
internal class SoftmaxWithLossD1 internal constructor(
    val outputSize: Int,
    val temperature: Float,
    val maskValue: Int? = null,
) : Output.D1() {
    override fun expect(input: Batch<IOType.D1>): Batch<IOType.D1> {
        val input = input / temperature
        val max = input.max()
        val exp = (input - max).exp()
        val sum = exp.sum()
        return exp / sum
    }

    override fun train(input: Batch<IOType.D1>, label: (Batch<IOType.D1>) -> Batch<IOType.D1>): TResult<IOType.D1> {
        val input = input / temperature
        val max = input.max()
        val exp = (input - max).exp()
        val sum = exp.sum()
        val output = exp / sum

        val label = label(output)

        val loss = -(output * label).sum()
            .ln(1e-7f)
            .batchAverage()
            .get()
        val delta = (output - label).let { diff ->
            if (maskValue == null) diff else diff * label.where(onTrue = 0f) { it eq maskValue.toFloat() }
        }
        return TResult(loss = loss, delta = delta)
    }
}

fun <T> NetworkBuilder.D1<T>.softmaxWithLoss(temperature: Float = 1f, maskValue: Int? = null) = addOutput(
    output = SoftmaxWithLossD1(
        outputSize = inputSize,
        temperature = temperature,
        maskValue = maskValue,
    ),
)

fun <I, O> NetworkBuilder.D1<I>.softmaxWithLoss(
    converter: NetworkBuilder.D1<I>.() -> Converter.D1<O>,
    temperature: Float = 1f,
    maskValue: Int? = null,
) = addOutput(
    output = SoftmaxWithLossD1(
        outputSize = inputSize,
        temperature = temperature,
        maskValue = maskValue,
    ),
    converter = converter,
)
