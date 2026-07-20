package com.wsr.knist.network.output.softmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.GraphBuilder
import com.wsr.knist.network.GraphScope.addOutput
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.converter.raw.RawD1
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.output.TResult
import kotlinx.serialization.Serializable

@Serializable
internal class SoftmaxWithLossD1 internal constructor(val temperature: Float) : Output.D1() {
    override fun IOScope.expect(input: Batch<IOType.D1>): Batch<IOType.D1> {
        val input = input / temperature
        val max = input.max()
        val exp = (input - max).exp()
        val sum = exp.sum()
        return exp / sum
    }

    override fun IOScope.train(
        input: Batch<IOType.D1>,
        label: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): TResult<IOType.D1> {
        val input = input / temperature
        val max = input.max()
        val exp = (input - max).exp()
        val sum = exp.sum()
        val output = exp / sum

        val label = label(output)

        val loss = 0f - (output * label).sum()
            .ln(1e-7f)
            .batchAverage()
        val delta = (output - label) / temperature
        return TResult(loss = loss, delta = delta)
    }
}

fun <I> NetworkBuilder.D1<I>.softmaxWithLoss(temperature: Float = 1f) = addOutput(
    output = SoftmaxWithLossD1(temperature = temperature),
    converter = { RawD1(inputI) },
)

fun <I, O> NetworkBuilder.D1<I>.softmaxWithLoss(
    temperature: Float = 1f,
    converter: NetworkBuilder.D1<I>.() -> Converter.D1<O>,
) = addOutput(
    output = SoftmaxWithLossD1(temperature = temperature),
    converter = converter,
)

fun GraphBuilder.D1.softmaxWithLoss(temperature: Float = 1f) = addOutput(
    output = SoftmaxWithLossD1(temperature = temperature),
    converter = RawD1(inputI),
)

fun <O> GraphBuilder.D1.softmaxWithLoss(
    temperature: Float = 1f,
    converter: GraphBuilder.D1.() -> Converter.D1<O>,
) = addOutput(
    output = SoftmaxWithLossD1(temperature = temperature),
    converter = converter(),
)
