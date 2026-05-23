package com.wsr.network.output.mean

import com.wsr.batch.Batch
import com.wsr.batch.reduction.average.batchAverage
import com.wsr.batch.elementwise.math.pow
import com.wsr.batch.elementwise.operation.minus.minus
import com.wsr.core.IOType
import com.wsr.core.reduction.average
import com.wsr.core.get
import com.wsr.core.elementwise.operation.times.times
import com.wsr.network.NetworkBuilder
import com.wsr.network.converter.Converter
import com.wsr.network.output.Output
import com.wsr.network.output.TResult
import kotlinx.serialization.Serializable

@Serializable
internal class MeanSquareD1 internal constructor(val outputSize: Int) : Output.D1() {
    override fun expect(input: Batch<IOType.D1>): Batch<IOType.D1> = input

    override fun train(input: Batch<IOType.D1>, label: (Batch<IOType.D1>) -> Batch<IOType.D1>): TResult<IOType.D1> {
        val delta = input - label(input)
        val loss = delta
            .pow(2)
            .batchAverage().average() * 0.5f
        return TResult(loss = loss.get(), delta = delta)
    }
}

fun <T> NetworkBuilder.D1<T>.meanSquare() = addOutput(MeanSquareD1(inputSize))

fun <I, O> NetworkBuilder.D1<I>.meanSquare(converter: NetworkBuilder.D1<I>.() -> Converter.D1<O>) = addOutput(
    output = MeanSquareD1(inputSize),
    converter = converter,
)
