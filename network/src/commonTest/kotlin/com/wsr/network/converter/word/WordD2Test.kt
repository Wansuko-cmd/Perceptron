@file:Suppress("NonAsciiCharacters")

package com.wsr.network.converter.word

import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.converter.word.WordD2
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class WordD2Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    private val words = listOf("<PAD>", "<UNK>", "hello", "world")
    private val length = 5
    private val unknownIndex = 1

    @Test
    fun `encode=単語をone-hotベクトルに変換`() {
        val target = WordD2(words = words, length = length, unknownIndex = unknownIndex)
        val input = listOf(
            listOf("hello", "world", "!"),
            listOf("new", "world", "<PAD>"),
        )

        val actual = target.encode(input)

        // hello world !
        assertEquals(expected = IOType.d1(0f, 0f, 1f, 0f), actual = actual[0][0])
        assertEquals(expected = IOType.d1(0f, 0f, 0f, 1f), actual = actual[0][1])
        assertEquals(expected = IOType.d1(0f, 1f, 0f, 0f), actual = actual[0][2])

        // new world <PAD>
        assertEquals(expected = IOType.d1(0f, 1f, 0f, 0f), actual = actual[1][0])
        assertEquals(expected = IOType.d1(0f, 0f, 0f, 1f), actual = actual[1][1])
        assertEquals(expected = IOType.d1(1f, 0f, 0f, 0f), actual = actual[1][2])
    }

    @Test
    fun `decode=one-hotベクトルを単語に変換`() {
        val target = WordD2(words = words, length = length, unknownIndex = unknownIndex)
        val input = batchOf(
            IOType.d2(
                IOType.d1(0f, 0f, 1f, 0f),
                IOType.d1(0f, 0f, 0f, 1f),
                IOType.d1(0f, 1f, 0f, 0f),
                IOType.d1(1f, 0f, 0f, 0f),
                IOType.d1(1f, 0f, 0f, 0f),
            ),
            IOType.d2(
                IOType.d1(0f, 1f, 0f, 0f),
                IOType.d1(0f, 0f, 0f, 1f),
                IOType.d1(1f, 0f, 0f, 0f),
                IOType.d1(1f, 0f, 0f, 0f),
                IOType.d1(1f, 0f, 0f, 0f),
            ),
        )

        val actual = target.decode(input)

        assertEquals(expected = listOf("hello", "world", "<UNK>", "<PAD>", "<PAD>"), actual = actual[0])
        assertEquals(expected = listOf("<UNK>", "world", "<PAD>", "<PAD>", "<PAD>"), actual = actual[1])
    }
}
