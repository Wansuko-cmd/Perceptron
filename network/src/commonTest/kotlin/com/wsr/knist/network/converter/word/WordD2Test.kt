@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.converter.word
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertContentEquals

class WordD2Test {
    private val words = listOf("<PAD>", "<UNK>", "hello", "world")
    private val length = 5
    private val unknownIndex = 1

    @Test
    fun `encode=単語をone-hotベクトルに変換`() = networkTestRule {
        val target = WordD2(words = words, length = length, unknownIndex = unknownIndex)
        val input = listOf(
            listOf("hello", "world", "!"),
            listOf("new", "world", "<PAD>"),
        )

        val actual = target.encode(input)

        // hello world !
        assertContentEquals(expected = IOType.d1(0f, 0f, 1f, 0f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 1f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, 1f, 0f, 0f), actual = actual[0][2])

        // new world <PAD>
        assertContentEquals(expected = IOType.d1(0f, 1f, 0f, 0f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 1f), actual = actual[1][1])
        assertContentEquals(expected = IOType.d1(1f, 0f, 0f, 0f), actual = actual[1][2])
    }

    @Test
    fun `decode=one-hotベクトルを単語に変換`() = networkTestRule {
        val target = WordD2(words = words, length = length, unknownIndex = unknownIndex)
        val input = Batch.of(
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

        assertContentEquals(expected = listOf("hello", "world", "<UNK>", "<PAD>", "<PAD>"), actual = actual[0])
        assertContentEquals(expected = listOf("<UNK>", "world", "<PAD>", "<PAD>", "<PAD>"), actual = actual[1])
    }
}
