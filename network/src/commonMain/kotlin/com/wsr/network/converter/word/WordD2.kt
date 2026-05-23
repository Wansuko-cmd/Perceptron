package com.wsr.network.converter.word

import com.wsr.base.data.DataBuffer
import com.wsr.batch.Batch
import com.wsr.batch.toList
import com.wsr.core.IOType
import com.wsr.core.reduction.maxIndex
import com.wsr.core.get
import com.wsr.core.set
import com.wsr.network.NetworkBuilder
import com.wsr.network.converter.Converter
import com.wsr.network.initializer.WeightInitializer
import com.wsr.network.optimizer.Optimizer
import kotlinx.serialization.Serializable

@Serializable
class WordD2(private val words: List<String>, private val length: Int, private val unknownIndex: Int) :
    Converter.D2<List<String>>() {
    override val outputX = length
    override val outputY = words.size
    private val wordToId = words.mapIndexed { index, word -> word to index }.toMap()

    override fun encode(input: List<List<String>>): Batch<IOType.D2> {
        val result = FloatArray(input.size * outputX * outputY)
        repeat(input.size) { b ->
            val offset = b * outputX * outputY
            val text = input[b]
            text.forEachIndexed { index, word ->
                val id = wordToId[word] ?: unknownIndex
                result[offset + index * outputY + id] = 1f
            }
            for (index in text.size until outputX) {
                result[offset + index * outputY] = 1f
            }
        }
        return Batch(size = input.size, shape = listOf(outputX, outputY), value = DataBuffer.create(result))
    }

    override fun decode(input: Batch<IOType.D2>): List<List<String>> = input.toList().map { input ->
        (0 until length).map { words[input[it].maxIndex()] }
    }
}

fun NetworkBuilder.Companion.wordD2(
    words: List<String>,
    length: Int,
    unknownIndex: Int,
    optimizer: Optimizer,
    initializer: WeightInitializer,
): NetworkBuilder.D2<List<String>> {
    check(unknownIndex in words.indices) { "unknownIndex must be within words range." }

    return inputD2(
        converter = WordD2(
            words = words,
            length = length,
            unknownIndex = unknownIndex,
        ),
        optimizer = optimizer,
        initializer = initializer,
    )
}
