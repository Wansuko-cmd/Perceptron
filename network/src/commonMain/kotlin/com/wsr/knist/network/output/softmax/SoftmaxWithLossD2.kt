package com.wsr.knist.network.output.softmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.math.ln
import com.wsr.knist.batch.elementwise.math.softmax
import com.wsr.knist.batch.elementwise.operation.div.div
import com.wsr.knist.batch.elementwise.operation.minus.minus
import com.wsr.knist.batch.elementwise.operation.times.times
import com.wsr.knist.batch.index.gather
import com.wsr.knist.batch.reduction.average.batchAverage
import com.wsr.knist.batch.reduction.sum
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.output.TResult
import kotlinx.serialization.Serializable

@Serializable
internal class SoftmaxWithLossD2 internal constructor(
    val outputX: Int,
    val outputY: Int,
    val temperature: Float,
    val maskValue: Int? = null,
) : Output.D2() {
    override fun expect(input: Batch<IOType.D2>): Batch<IOType.D2> {
        val input = input / temperature
        return input.softmax(axis = 1)
    }

    override fun train(input: Batch<IOType.D2>, label: (Batch<IOType.D2>) -> Batch<IOType.D2>): TResult<IOType.D2> {
        val input = input / temperature
        val output = input.softmax(axis = 1)

        val label = label(output)
        val mask = label.generateMask()

        // -log(p)
        val losses = -1f * (output * label).sum(axis = 1).ln(1e-7f)
        val maskD1 = mask.sum(axis = 1)
        val maskedLosses = losses * maskD1

        // 有効値のみの平均を取る
        val loss = (maskedLosses.sum() / maskD1.sum()).batchAverage().get()

        val delta = (output - label) * mask
        return TResult(loss = loss, delta = delta)
    }

    private fun Batch<IOType.D2>.generateMask(): Batch<IOType.D2> = when {
        maskValue == null -> Batch(size) { IOType.d2(shape = shape) { _, _ -> 1f } }

        else -> {
            IOType.d0(maskValue.toFloat())
                .gather(other = this, axis = 1)
                .gather(
                    other = IOType.d2(
                        IOType.d1(outputY) { 1f },
                        IOType.d1(outputY) { 0f },
                    ),
                )
        }
    }
}

fun <T> NetworkBuilder.D2<T>.softmaxWithLoss(temperature: Float = 1f, maskValue: Int? = null) = addOutput(
    output = SoftmaxWithLossD2(
        outputX = inputX,
        outputY = inputY,
        temperature = temperature,
        maskValue = maskValue,
    ),
)

fun <I, O> NetworkBuilder.D2<I>.softmaxWithLoss(
    converter: NetworkBuilder.D2<I>.() -> Converter.D2<O>,
    temperature: Float = 1f,
    maskValue: Int? = null,
) = addOutput(
    output = SoftmaxWithLossD2(
        outputX = inputX,
        outputY = inputY,
        temperature = temperature,
        maskValue = maskValue,
    ),
    converter = converter,
)
