@file:Suppress("NonAsciiCharacters")

package com.wsr.network.converter.char

import com.wsr.batch.batchOf
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.network.assertContentEquals
import com.wsr.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class CharsD1Test {
    @Test
    fun `encode=文字列を文字IDベクトルに変換`() = networkTestRule {
        val target = CharsD1(outputSize = 5)
        val input = listOf("boo")

        val actual = target.encode(input)

        assertContentEquals(
            expected = batchOf(IOType.d1(2f, 15f, 15f, 0f, 0f)),
            actual = actual,
        )
    }

    @Test
    fun `decode=文字IDベクトルを文字列に変換`() = networkTestRule {
        val target = CharsD1(outputSize = 5)
        val input = batchOf(IOType.d1(2f, 15f, 15f))

        val actual = target.decode(input)

        assertEquals(expected = listOf("boo"), actual = actual)
    }
}
