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
internal class MeanSquareD2 internal constructor(val outputX: Int, val outputY: Int) : Output.D2() {
    override fun expect(input: Batch<IOType.D2>): Batch<IOType.D2> = input

    override fun train(input: Batch<IOType.D2>, label: (Batch<IOType.D2>) -> Batch<IOType.D2>): TResult<IOType.D2> {
        val delta = input - label(input)
        val loss = delta
            .pow(2)
            .batchAverage().average() * 0.5f
        return TResult(loss = loss.unwrap(), delta = delta)
    }
}

fun <T> NetworkBuilder.D2<T>.meanSquare() = addOutput(
    output = MeanSquareD2(
        outputX = inputX,
        outputY = inputY,
    ),
)

fun <I, O> NetworkBuilder.D2<I>.meanSquare(converter: NetworkBuilder.D2<I>.() -> Converter.D2<O>) = addOutput(
    output = MeanSquareD2(
        outputX = inputX,
        outputY = inputY,
    ),
    converter = converter,
)
