@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.collection

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test
import kotlin.test.assertEquals

class MinTest {
    @Test
    fun `1次元内最小値`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.min(x = x)

        assertEquals(expected = 0f, actual = actual)
    }

    @Test
    fun `2次元最小値_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.min(x = x, xi = 4, xj = 6, axis = 0)

        assertEquals(expected = DataBuffer.create(0f, 1f, 2f, 3f, 4f, 5f), actual = actual)
    }

    @Test
    fun `2次元最小値_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.min(x = x, xi = 4, xj = 6, axis = 1)

        assertEquals(expected = DataBuffer.create(0f, 6f, 12f, 18f), actual = actual)
    }

    @Test
    fun `3次元最小値_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.min(x = x, xi = 4, xj = 3, xk = 2, axis = 0)

        assertEquals(expected = DataBuffer.create(0f, 1f, 2f, 3f, 4f, 5f), actual = actual)
    }

    @Test
    fun `3次元最小値_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.min(x = x, xi = 4, xj = 3, xk = 2, axis = 1)

        assertEquals(expected = DataBuffer.create(0f, 1f, 6f, 7f, 12f, 13f, 18f, 19f), actual = actual)
    }

    @Test
    fun `3次元最小値_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.min(x = x, xi = 4, xj = 3, xk = 2, axis = 2)

        assertEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f, 18f, 20f, 22f),
            actual = actual,
        )
    }
}
