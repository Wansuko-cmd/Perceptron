@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.buffer

import com.wsr.base.data.DataBuffer
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ContentEqualsTest {
    @Test
    fun `contentEquals=要素数と中身が同じならtrue`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })
        val other = DataBuffer.create(FloatArray(10) { it.toFloat() })

        val actual = input.contentEquals(other)

        assertTrue(actual)
    }

    @Test
    fun `contentEquals=要素数が違うならfalse`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })
        val other = DataBuffer.create(FloatArray(9) { it.toFloat() })

        val actual = input.contentEquals(other)

        assertFalse(actual)
    }

    @Test
    fun `contentEquals=中身が違うならfalse`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(10) { it.toFloat() })
        val other = DataBuffer.create(FloatArray(10) { it.toFloat() + 1 })

        val actual = input.contentEquals(other)

        assertFalse(actual)
    }
}
