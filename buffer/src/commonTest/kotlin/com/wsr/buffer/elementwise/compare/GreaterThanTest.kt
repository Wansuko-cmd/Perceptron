@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.elementwise.compare

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import kotlin.test.Test

class GreaterThanTest {
    @Test
    fun `greaterThan=入力のFloat値より大きいかどうか`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })

        val actual = Backend.greaterThan(input, 4f)

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(0f, 0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f, 1f)),
            actual = actual,
        )
    }

    @Test
    fun `greaterThan=入力のDataBufferより大きいかどうか`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })
        val other = DataBuffer.create(FloatArray(10) { 4f })

        val actual = Backend.greaterThan(input, other)

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(0f, 0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f, 1f)),
            actual = actual,
        )
    }
}
