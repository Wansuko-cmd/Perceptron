@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.converter.char
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class CharsD1Test {
    @Test
    fun `encode=文字列を文字IDベクトルに変換`() = networkTestRule {
        val target = CharsD1(outputSize = 5)
        val input = listOf("boo")

        val actual = target.encode(input)

        assertContentEquals(
            expected = Batch.of(IOType.d1(2f, 15f, 15f, 0f, 0f)),
            actual = actual,
        )
    }

    @Test
    fun `decode=文字IDベクトルを文字列に変換`() = networkTestRule {
        val target = CharsD1(outputSize = 5)
        val input = Batch.of(IOType.d1(2f, 15f, 15f))

        val actual = target.decode(input)

        assertEquals(expected = listOf("boo"), actual = actual)
    }
}
