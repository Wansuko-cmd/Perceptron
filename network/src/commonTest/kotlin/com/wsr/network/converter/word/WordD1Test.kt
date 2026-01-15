@file:Suppress("NonAsciiCharacters")

package com.wsr.network.converter.word

import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class WordD1Test {
    private val words = listOf("<PAD>", "<UNK>", "hello", "world")
    private val unknownIndex = 1

    @Test
    fun `encode=単語をone-hotベクトルに変換`() = networkTestRule {
        val target = WordD1(words = words, unknownIndex = unknownIndex)
        val input = listOf("hello", "world", "!")

        val actual = target.encode(input)

        assertEquals(
            expected = IOType.d1(0f, 0f, 1f, 0f),
            actual = actual[0],
        )
        assertEquals(
            expected = IOType.d1(0f, 0f, 0f, 1f),
            actual = actual[1],
        )
        assertEquals(
            expected = IOType.d1(0f, 1f, 0f, 0f),
            actual = actual[2],
        )
    }

    @Test
    fun `decode=one-hotベクトルを単語に変換`() = networkTestRule {
        val target = WordD1(words = words, unknownIndex = unknownIndex)
        val input = batchOf(
            IOType.d1(listOf(0f, 0f, 1f, 0f)),
            IOType.d1(listOf(0f, 0f, 0f, 1f)),
            IOType.d1(listOf(0f, 1f, 0f, 0f)),
        )

        val actual = target.decode(input)

        assertEquals(expected = listOf("hello", "world", "<UNK>"), actual = actual)
    }
}
