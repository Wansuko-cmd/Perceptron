package com.wsr.knist.network.output.sigmoid

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d2
import com.wsr.knist.batch.elementwise.math.ln
import com.wsr.knist.batch.elementwise.math.sigmoid
import com.wsr.knist.batch.elementwise.operation.minus.minus
import com.wsr.knist.batch.elementwise.operation.plus.plus
import com.wsr.knist.batch.elementwise.operation.times.times
import com.wsr.knist.batch.reduction.average.batchAverage
import com.wsr.knist.batch.reduction.sum
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.operation.minus.minus
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.output.TResult
import kotlinx.serialization.Serializable

@Serializable
internal class SigmoidWithLossD2 internal constructor(val outputX: Int, val outputY: Int) : Output.D2() {
    override fun IOScope.expect(input: Batch<IOType.D2>): Batch<IOType.D2> = input.sigmoid()

    override fun IOScope.train(input: Batch<IOType.D2>, label: (Batch<IOType.D2>) -> Batch<IOType.D2>): TResult<IOType.D2> {
        val output = input.sigmoid()
        val label = label(output)
        val one = Batch.d2(label.size, outputX, outputY) { _, _ -> 1f }
        val loss = run {
            val y = label * output.ln(1e-7f)
            val p = (one - label) * (one - output).ln(1e-7f)
            0f - (y + p).sum().batchAverage()
        }
        val delta = output - label
        return TResult(loss = loss, delta = delta)
    }
}

fun <T> NetworkBuilder.D2<T>.sigmoidWithLoss() = addOutput(
    output = SigmoidWithLossD2(
        outputX = inputX,
        outputY = inputY,
    ),
)

fun <I, O> NetworkBuilder.D2<I>.sigmoidWithLoss(converter: NetworkBuilder.D2<I>.() -> Converter.D2<O>) = addOutput(
    output = SigmoidWithLossD2(
        outputX = inputX,
        outputY = inputY,
    ),
    converter = converter,
)
