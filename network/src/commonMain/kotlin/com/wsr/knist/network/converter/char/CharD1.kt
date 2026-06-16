package com.wsr.knist.network.converter.char

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.reduction.maxIndex
import com.wsr.knist.batch.shape.toBatch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.set
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.converter.Converter
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import kotlinx.serialization.Serializable

@Serializable
class CharD1 : Converter.D1<Char>() {
    override val outputSize = chars.size
    override fun encode(input: List<Char>): Batch<IOType.D1> = input
        .map { char ->
            val id = charToId[char] ?: 0
            IOType.d1(outputSize).also { it[id] = 1f }
        }.toBatch()

    override fun decode(input: Batch<IOType.D1>): List<Char> = input
        .maxIndex()
        .value
        .toFloatArray()
        .map { chars[it.toInt()] }

    companion object Companion {
        private val chars = " abcdefghijklmnopqrstuvwxyz.,!?".toList()
        private val charToId = chars.mapIndexed { index, char -> char to index }.toMap()
        val vocabSize = chars.size
    }
}

fun NetworkBuilder.Companion.charD1(optimizer: Optimizer, initializer: WeightInitializer) = inputD1(
    converter = CharD1(),
    optimizer = optimizer,
    initializer = initializer,
)
