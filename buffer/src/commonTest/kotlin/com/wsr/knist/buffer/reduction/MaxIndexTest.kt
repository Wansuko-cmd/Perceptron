@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.buffer.reduction

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.buffer.assertContentEquals
import com.wsr.knist.buffer.bufferTestRule
import kotlin.test.Test

class MaxIndexTest {
    @Test
    fun `1次元最大値インデックス`() = bufferTestRule {
        val x = DataBuffer.create(1f, 4f, 2f, 3f)

        val actual = Backend.maxIndex(x = x)

        assertContentEquals(expected = DataBuffer.create(1f), actual = actual)
    }

    @Test
    fun `2次元最大値インデックス_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.maxIndex(x = x, xi = 4, xj = 6, axis = 0)

        assertContentEquals(expected = DataBuffer.create(3f, 3f, 3f, 3f, 3f, 3f), actual = actual)
    }

    @Test
    fun `2次元最大値インデックス_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.maxIndex(x = x, xi = 4, xj = 6, axis = 1)

        assertContentEquals(expected = DataBuffer.create(5f, 5f, 5f, 5f), actual = actual)
    }

    @Test
    fun `3次元最大値インデックス_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.maxIndex(x = x, xi = 4, xj = 3, xk = 2, axis = 0)

        assertContentEquals(expected = DataBuffer.create(3f, 3f, 3f, 3f, 3f, 3f), actual = actual)
    }

    @Test
    fun `3次元最大値インデックス_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.maxIndex(x = x, xi = 4, xj = 3, xk = 2, axis = 1)

        assertContentEquals(expected = DataBuffer.create(2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f), actual = actual)
    }

    @Test
    fun `3次元最大値インデックス_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.maxIndex(x = x, xi = 4, xj = 3, xk = 2, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(FloatArray(12) { 1f }),
            actual = actual,
        )
    }
}
