package dataset.mnist

import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.reduction.maxIndex
import com.wsr.knist.core.IOType
import com.wsr.knist.network.converter.Converter
import kotlinx.serialization.Serializable

@Serializable
data class PixelConverter(override val outputI: Int, override val outputJ: Int) : Converter.D2<List<List<Float>>>() {
    override fun encode(input: List<List<Float>>): Batch<IOType.D2> {
        val value = input.flatten().toFloatArray()
        return Batch(
            size = input.size,
            shape = listOf(outputI, outputJ),
            value = DataBuffer.create(value),
        )
    }

    override fun decode(input: Batch<IOType.D2>): List<List<Float>> = input.value.toFloatArray()
        .toList()
        .chunked(input.size)
}

@Serializable
data class LabelConverter(override val outputI: Int) : Converter.D1<List<Int>>() {
    override fun encode(input: List<Int>): Batch<IOType.D1> {
        val value = FloatArray(input.size * outputI)
        repeat(input.size) { batchIndex ->
            val label = input[batchIndex]
            value[batchIndex * outputI + label] = 1f
        }
        return Batch(size = input.size, shape = listOf(outputI), value = DataBuffer.create(value))
    }

    override fun decode(input: Batch<IOType.D1>): List<Int> = input.maxIndex()
        .value
        .toFloatArray()
        .map { it.toInt() }
}
