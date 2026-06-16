package com.wsr.knist.network.converter.word

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.shape.toBatch
import com.wsr.knist.batch.shape.toList
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import kotlinx.serialization.Serializable

@Serializable
class WordsD1(
    override val outputSize: Int,
    private val words: List<String>,
    private val unknownIndex: Int,
    private val paddingIndex: Int,
) : Converter.D1<List<String>>() {
    val vocabSize = words.size
    private val wordToId = words.mapIndexed { index, word -> word to index.toFloat() }.toMap()

    override fun IOScope.encode(input: List<List<String>>): Batch<IOType.D1> = input.map { sentence ->
        val tokenIds = sentence
            .take(outputSize)
            .map { wordToId[it] ?: unknownIndex.toFloat() }

        FloatArray(outputSize) { paddingIndex.toFloat() }
            .apply { tokenIds.toFloatArray().copyInto(this) }
            .let { IOType.d1(it) }
    }.toBatch()

    override fun IOScope.decode(input: Batch<IOType.D1>): List<List<String>> = input.toList().map { input ->
        input.value.toFloatArray()
            .toList()
            .mapNotNull { id ->
                val index = id.toInt()
                if (index == paddingIndex) null else words.getOrNull(index)
            }
    }
}

fun NetworkBuilder.Companion.wordsD1(
    maxLength: Int,
    words: List<String>,
    unknownIndex: Int,
    paddingIndex: Int,
    optimizer: Optimizer,
    initializer: WeightInitializer,
): NetworkBuilder.D1<List<String>> {
    check(unknownIndex in words.indices) { "unknownIndex must be within words range." }
    check(paddingIndex in words.indices) { "paddingIndex must be within words range." }

    return inputD1(
        converter = WordsD1(
            outputSize = maxLength,
            words = words,
            unknownIndex = unknownIndex,
            paddingIndex = paddingIndex,
        ),
        optimizer = optimizer,
        initializer = initializer,
    )
}
