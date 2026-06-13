package dataset.mnist

import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.reduction.maxIndex
import com.wsr.knist.batch.shape.toList
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import kotlinx.serialization.Serializable

@Serializable
data class PixelConverter(override val outputX: Int, override val outputY: Int) : Converter.D2<List<Float>>() {
    override fun encode(input: List<List<Float>>): Batch<IOType.D2> {
        val value = input.flatten().toFloatArray()
        return Batch(
            size = input.size,
            shape = listOf(outputX, outputY),
            value = DataBuffer.create(value),
        )
    }

    override fun decode(input: Batch<IOType.D2>): List<List<Float>> = input.value.toFloatArray()
        .toList()
        .chunked(input.size)
}

fun NetworkBuilder.Companion.inputPx(x: Int, y: Int, optimizer: Optimizer, initializer: WeightInitializer) = inputD2(
    converter = PixelConverter(x, y),
    optimizer = optimizer,
    initializer = initializer,
)

@Serializable
data class LabelConverter(override val outputSize: Int) : Converter.D1<Int>() {
    override fun encode(input: List<Int>): Batch<IOType.D1> {
        val value = FloatArray(input.size * outputSize)
        repeat(input.size) { batchIndex ->
            val label = input[batchIndex]
            value[batchIndex * outputSize + label] = 1f
        }
        return Batch(size = input.size, shape = listOf(outputSize), value = DataBuffer.create(value))
    }

    override fun decode(input: Batch<IOType.D1>): List<Int> = input.maxIndex()
        .value
        .toFloatArray()
        .map { it.toInt() }
}
