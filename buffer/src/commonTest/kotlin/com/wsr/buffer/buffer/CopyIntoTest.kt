@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.buffer

import com.wsr.base.data.DataBuffer
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test
import kotlin.test.assertEquals

class CopyIntoTest {
    @Test
    fun `指定範囲のコピー`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })
        val actual = DataBuffer.create(FloatArray(15))

        input.copyInto(actual, 5)

        assertEquals(
            expected = 15,
            actual = actual.size,
        )
        assertEquals(
            expected = DataBuffer.create(
                floatArrayOf(
                    0f, 0f, 0f, 0f, 0f,
                    0f, 1f, 2f, 3f, 4f,
                    5f, 6f, 7f, 8f, 9f,
                ),
            ),
            actual = actual,
        )
    }
}
