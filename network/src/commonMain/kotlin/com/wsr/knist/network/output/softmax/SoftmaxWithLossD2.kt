package com.wsr.knist.network.output.softmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.converter.raw.RawD2
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.output.TResult
import kotlinx.serialization.Serializable

@Serializable
internal class SoftmaxWithLossD2 internal constructor(
    val outputJ: Int,
    val temperature: Float,
    val maskValue: Int? = null,
) : Output.D2() {
    private val maskBool: IOType.D2.Global by lazy {
        IOType.d2(
            IOType.d1(outputJ) { 1f },
            IOType.d1(outputJ) { 0f },
        )
    }

    override fun IOScope.expect(input: Batch<IOType.D2>): Batch<IOType.D2> {
        val input = input / temperature
        return input.softmax(axis = 1)
    }

    override fun IOScope.train(
        input: Batch<IOType.D2>,
        label: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): TResult<IOType.D2> {
        val input = input / temperature
        val output = input.softmax(axis = 1)

        val label = label(output)
        val mask = when {
            maskValue == null -> Batch.d2(label.size, label.shape) { _, _ -> 1f }

            else -> {
                IOType.d0(maskValue.toFloat())
                    .gather(other = label, axis = 1)
                    .gather(other = maskBool)
            }
        }

        // -log(p)
        val losses = -1f * (output * label).sum(axis = 1).ln(1e-7f)
        val maskD1 = mask.sum(axis = 1)
        val maskedLosses = losses * maskD1

        // 有効値のみの平均を取る
        val loss = (maskedLosses.sum() / maskD1.sum()).batchAverage()

        val delta = (output - label) * mask
        return TResult(loss = loss, delta = delta)
    }
}

fun <I> NetworkBuilder.D2<I>.softmaxWithLoss(
    temperature: Float = 1f,
    maskValue: Int? = null,
) = addOutput(
    output = SoftmaxWithLossD2(
        outputJ = inputJ,
        temperature = temperature,
        maskValue = maskValue,
    ),
    converter = { RawD2(inputI, inputJ) },
)


fun <I, O> NetworkBuilder.D2<I>.softmaxWithLoss(
    converter: NetworkBuilder.D2<I>.() -> Converter.D2<O>,
    temperature: Float = 1f,
    maskValue: Int? = null,
) = addOutput(
    output = SoftmaxWithLossD2(
        outputJ = inputJ,
        temperature = temperature,
        maskValue = maskValue,
    ),
    converter = converter,
)
