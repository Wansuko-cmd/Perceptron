package com.wsr.knist.network.converter.char

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
class CharsD1(override val outputSize: Int) : Converter.D1<String>() {
    override fun IOScope.encode(input: List<String>): Batch<IOType.D1> = input.toList().map { text ->
        IOType.d1(outputSize) { index ->
            text.getOrNull(index)
                ?.let { charToId[it] }
                ?: 0f
        }
    }.toBatch()

    override fun IOScope.decode(input: Batch<IOType.D1>): List<String> = input.toList().map { input ->
        input.value.toFloatArray()
            .map { chars.getOrNull(it.toInt()) }
            .filterNotNull()
            .joinToString("")
    }

    companion object Companion {
        private val chars = " abcdefghijklmnopqrstuvwxyz.,!?".toList()
        private val charToId = chars.mapIndexed { index, char -> char to index.toFloat() }.toMap()
        val vocabSize = chars.size
    }
}

fun NetworkBuilder.Companion.charsD1(maxLength: Int, optimizer: Optimizer, initializer: WeightInitializer) = inputD1(
    converter = CharsD1(maxLength),
    optimizer = optimizer,
    initializer = initializer,
)
