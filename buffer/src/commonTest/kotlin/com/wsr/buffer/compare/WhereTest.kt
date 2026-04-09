@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.compare

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import kotlin.test.Test

class WhereTest {
    @Test
    fun `Float_where_Float=conditionが1の時x_0のときy`() = bufferTestRule {
        val condition = DataBuffer.create(floatArrayOf(0f, 1f, 0f, 1f, 0f))
        val x = 2f
        val y = 3f

        val actual = Backend.where(condition, x, y)

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(3f, 2f, 3f, 2f, 3f)),
            actual = actual,
        )
    }

    @Test
    fun `Float_where_Buffer=conditionが1の時x_0のときy`() = bufferTestRule {
        val condition = DataBuffer.create(floatArrayOf(0f, 1f, 0f, 1f, 0f))
        val x = 2f
        val y = DataBuffer.create(FloatArray(5) { 2f * it + 1 })

        val actual = Backend.where(condition, x, y)

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(1f, 2f, 5f, 2f, 9f)),
            actual = actual,
        )
    }

    @Test
    fun `Buffer_where_Float=conditionが1の時x_0のときy`() = bufferTestRule {
        val condition = DataBuffer.create(floatArrayOf(0f, 1f, 0f, 1f, 0f))
        val x = DataBuffer.create(FloatArray(5) { 2f * it })
        val y = 3f

        val actual = Backend.where(condition, x, y)

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(3f, 2f, 3f, 6f, 3f)),
            actual = actual,
        )
    }

    @Test
    fun `Buffer_where_Buffer=conditionが1の時x_0のときy`() = bufferTestRule {
        val condition = DataBuffer.create(floatArrayOf(0f, 1f, 0f, 1f, 0f))
        val x = DataBuffer.create(FloatArray(5) { 2f * it })
        val y = DataBuffer.create(FloatArray(5) { 2f * it + 1 })

        val actual = Backend.where(condition, x, y)

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(1f, 2f, 5f, 6f, 9f)),
            actual = actual,
        )
    }
}
