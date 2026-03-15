@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.buffer

import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test
import kotlin.test.assertEquals

class CopyIntoTest {
    @Test
    fun `copyInto=指定箇所にコピー`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })
        val actual = DataBuffer.create(FloatArray(15))

        input.copyInto(dest = actual, destIndices = 5 until 10)

        assertEquals(
            expected = 15,
            actual = actual.size,
        )
        assertContentEquals(
            expected = DataBuffer.create(
                floatArrayOf(
                    0f, 0f, 0f, 0f, 0f,
                    0f, 1f, 2f, 3f, 4f,
                    0f, 0f, 0f, 0f, 0f,
                ),
            ),
            actual = actual,
        )
    }

    @Test
    fun `copyInto=step飛ばしでの指定も可能`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })
        val actual = DataBuffer.create(FloatArray(15))

        input.copyInto(dest = actual, destIndices = 5 until 15 step 3)

        assertEquals(
            expected = 15,
            actual = actual.size,
        )
        assertContentEquals(
            expected = DataBuffer.create(
                floatArrayOf(
                    0f, 0f, 0f, 0f, 0f,
                    0f, 0f, 0f, 1f, 0f,
                    0f, 2f, 0f, 0f, 3f,
                ),
            ),
            actual = actual,
        )
    }

    @Test
    fun `copyInto=downToでの指定も可能`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })
        val actual = DataBuffer.create(FloatArray(15))

        input.copyInto(dest = actual, destIndices = 14 downTo 5 step 2)

        assertEquals(
            expected = 15,
            actual = actual.size,
        )
        assertContentEquals(
            expected = DataBuffer.create(
                floatArrayOf(
                    0f, 0f, 0f, 0f, 0f,
                    0f, 4f, 0f, 3f, 0f,
                    2f, 0f, 1f, 0f, 0f,
                ),
            ),
            actual = actual,
        )
    }
}
