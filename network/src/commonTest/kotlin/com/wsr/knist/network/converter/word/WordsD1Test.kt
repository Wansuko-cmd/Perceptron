@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.converter.word
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertContentEquals

class WordsD1Test {
    private val words = listOf("<PAD>", "<UNK>", "hello", "world")
    private val outputSize = 5
    private val paddingIndex = 0
    private val unknownIndex = 1

    @Test
    fun `encode=単語列を単語IDベクトルに変換`() = networkTestRule {
        val target = WordsD1(
            words = words,
            outputI = outputSize,
            paddingIndex = paddingIndex,
            unknownIndex = unknownIndex,
        )
        val input = listOf(
            listOf("hello", "world", "!"),
            listOf("new", "world", "<PAD>"),
        )

        val actual = target.encode(input)

        assertContentEquals(
            expected = IOType.d1(2f, 3f, 1f, 0f, 0f),
            actual = actual[0],
        )
        assertContentEquals(
            expected = IOType.d1(1f, 3f, 0f, 0f, 0f),
            actual = actual[1],
        )
    }

    @Test
    fun `decode=単語IDベクトルを単語列に変換`() = networkTestRule {
        val target = WordsD1(
            words = words,
            outputI = outputSize,
            paddingIndex = paddingIndex,
            unknownIndex = unknownIndex,
        )
        val input = Batch.of(
            IOType.d1(2f, 3f, 1f, 0f, 0f),
            IOType.d1(1f, 3f, 0f, 0f, 0f),
        )

        val actual = target.decode(input)

        assertContentEquals(expected = listOf("hello", "world", "<UNK>"), actual = actual[0])
        assertContentEquals(expected = listOf("<UNK>", "world"), actual = actual[1])
    }
}
