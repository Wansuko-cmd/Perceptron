@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.buffer

import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test
import kotlin.test.assertEquals

class SliceTest {
    @Test
    fun `slice=指定範囲の切り取り`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = input.slice(5 until 15)

        assertEquals(
            expected = 10,
            actual = actual.size,
        )
        assertContentEquals(
            expected = DataBuffer.create(FloatArray(10) { it + 5f }),
            actual = actual,
        )
    }

    @Test
    fun `slice=stepの指定が可能`() = bufferTestRule {
        val input = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = input.slice(5 until 15 step 2)

        assertEquals(
            expected = 5,
            actual = actual.size,
        )
        assertContentEquals(
            expected = DataBuffer.create(FloatArray(5) { it * 2 + 5f }),
            actual = actual,
        )
    }
}
