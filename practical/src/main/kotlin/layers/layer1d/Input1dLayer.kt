package layers.layer1d

import exception.DomainException
import layers.IOType
import layers.Layer
import kotlin.random.Random

data class Input1dLayer(val channel: Int, val inputSize: Int) : Layer<IOType.IOType1d> {
    override val activationFunction: (Double) -> Double = { throw DomainException.UnreachableCodeException() }
    override fun forward(
        input: IOType,
        output: IOType,
        weight: Array<IOType>,
    ) = throw DomainException.UnreachableCodeException()
    override fun calcDelta(
        beforeDelta: DoubleArray,
        beforeOutput: IOType,
        delta: DoubleArray,
        weight: Array<IOType>,
    ) = throw DomainException.UnreachableCodeException()
    override fun backward(
        weight: Array<IOType>,
        delta: DoubleArray,
        input: IOType,
        rate: Double,
    ) = throw DomainException.UnreachableCodeException()
    override fun createWeight(input: IOType, random: Random): Array<IOType> =
        throw DomainException.UnreachableCodeException()
    override fun createOutput(input: IOType): IOType.IOType1d =
        IOType.IOType1d(Array(channel) { DoubleArray(inputSize) { 0.0 } })

    // 実際にこのdelta配列が使われることはない
    override fun createDelta(input: IOType): DoubleArray =
        DoubleArray(input.asIOType0d().value.size) { 0.0 }
}
