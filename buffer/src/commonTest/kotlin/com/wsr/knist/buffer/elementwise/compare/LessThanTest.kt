@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.buffer.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.buffer.assertContentEquals
import com.wsr.knist.buffer.bufferTestRule
import kotlin.test.Test

class LessThanTest {
    @Test
    fun `lessThan=入力のFloat値より大きいかどうか`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })

        val actual = Backend.lessThan(input, 5f)

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(1f, 1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f, 0f)),
            actual = actual,
        )
    }

    @Test
    fun `lessThan=入力のDataBufferより大きいかどうか`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })
        val other = DataBuffer.create(FloatArray(10) { 5f })

        val actual = Backend.lessThan(input, other)

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(1f, 1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f, 0f)),
            actual = actual,
        )
    }
}
