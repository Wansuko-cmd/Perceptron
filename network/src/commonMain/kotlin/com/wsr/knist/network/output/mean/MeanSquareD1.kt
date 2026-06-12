package com.wsr.knist.network.output.mean

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.math.pow
import com.wsr.knist.batch.elementwise.operation.minus.minus
import com.wsr.knist.batch.reduction.average.batchAverage
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.operation.times.times
import com.wsr.knist.core.reduction.average
import com.wsr.knist.core.unwrap
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.output.Output
import com.wsr.knist.network.output.TResult
import kotlinx.serialization.Serializable

@Serializable
internal class MeanSquareD1 internal constructor(val outputSize: Int) : Output.D1() {
    override fun expect(input: Batch<IOType.D1>): Batch<IOType.D1> = input

    override fun train(input: Batch<IOType.D1>, label: (Batch<IOType.D1>) -> Batch<IOType.D1>): TResult<IOType.D1> {
        val delta = input - label(input)
        val loss = delta
            .pow(2)
            .batchAverage().average() * 0.5f
        return TResult(loss = loss.unwrap(), delta = delta)
    }
}

fun <T> NetworkBuilder.D1<T>.meanSquare() = addOutput(MeanSquareD1(inputSize))

fun <I, O> NetworkBuilder.D1<I>.meanSquare(converter: NetworkBuilder.D1<I>.() -> Converter.D1<O>) = addOutput(
    output = MeanSquareD1(inputSize),
    converter = converter,
)
