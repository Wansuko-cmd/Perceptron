@file:Suppress("NonAsciiCharacters")

package com.wsr.network.converter.word

import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.converter.word.WordsD1
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.network.NetworkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class WordsD1Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    private val words = listOf("<PAD>", "<UNK>", "hello", "world")
    private val outputSize = 5
    private val paddingIndex = 0
    private val unknownIndex = 1

    @Test
    fun `encode=単語列を単語IDベクトルに変換`() {
        val target = WordsD1(
            words = words,
            outputSize = outputSize,
            paddingIndex = paddingIndex,
            unknownIndex = unknownIndex,
        )
        val input = listOf(
            listOf("hello", "world", "!"),
            listOf("new", "world", "<PAD>"),
        )

        val actual = target.encode(input)

        assertEquals(
            expected = IOType.d1(2f, 3f, 1f, 0f, 0f),
            actual = actual[0],
        )
        assertEquals(
            expected = IOType.d1(1f, 3f, 0f, 0f, 0f),
            actual = actual[1],
        )
    }

    @Test
    fun `decode=単語IDベクトルを単語列に変換`() {
        val target = WordsD1(
            words = words,
            outputSize = outputSize,
            paddingIndex = paddingIndex,
            unknownIndex = unknownIndex,
        )
        val input = batchOf(
            IOType.d1(2f, 3f, 1f, 0f, 0f),
            IOType.d1(1f, 3f, 0f, 0f, 0f),
        )

        val actual = target.decode(input)

        assertEquals(expected = listOf("hello", "world", "<UNK>"), actual = actual[0])
        assertEquals(expected = listOf("<UNK>", "world"), actual = actual[1])
    }
}
