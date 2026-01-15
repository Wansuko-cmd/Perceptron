@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.collection

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test
import kotlin.test.assertEquals

class MaxTest {
    @Test
    fun `1次元内最大値`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.max(x = x)

        assertEquals(expected = 23f, actual = actual)
    }

    @Test
    fun `2次元最大値_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.max(x = x, xi = 4, xj = 6, axis = 0)

        assertEquals(expected = DataBuffer.create(18f, 19f, 20f, 21f, 22f, 23f), actual = actual)
    }

    @Test
    fun `2次元最大値_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.max(x = x, xi = 4, xj = 6, axis = 1)

        assertEquals(expected = DataBuffer.create(5f, 11f, 17f, 23f), actual = actual)
    }

    @Test
    fun `3次元最大値_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.max(x = x, xi = 4, xj = 3, xk = 2, axis = 0)

        assertEquals(expected = DataBuffer.create(18f, 19f, 20f, 21f, 22f, 23f), actual = actual)
    }

    @Test
    fun `3次元最大値_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.max(x = x, xi = 4, xj = 3, xk = 2, axis = 1)

        assertEquals(expected = DataBuffer.create(4f, 5f, 10f, 11f, 16f, 17f, 22f, 23f), actual = actual)
    }

    @Test
    fun `3次元最大値_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.max(x = x, xi = 4, xj = 3, xk = 2, axis = 2)

        assertEquals(
            expected = DataBuffer.create(1f, 3f, 5f, 7f, 9f, 11f, 13f, 15f, 17f, 19f, 21f, 23f),
            actual = actual,
        )
    }

    @Test
    fun `4次元最大値_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.max(x = x, xi = 2, xj = 2, xk = 2, xl = 3, axis = 0)

        assertEquals(
            expected = DataBuffer.create(12f, 13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f, 21f, 22f, 23f),
            actual = actual,
        )
    }

    @Test
    fun `4次元最大値_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.max(x = x, xi = 2, xj = 2, xk = 2, xl = 3, axis = 1)

        assertEquals(
            expected = DataBuffer.create(6f, 7f, 8f, 9f, 10f, 11f, 18f, 19f, 20f, 21f, 22f, 23f),
            actual = actual,
        )
    }

    @Test
    fun `4次元最大値_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.max(x = x, xi = 2, xj = 2, xk = 2, xl = 3, axis = 2)

        assertEquals(
            expected = DataBuffer.create(3f, 4f, 5f, 9f, 10f, 11f, 15f, 16f, 17f, 21f, 22f, 23f),
            actual = actual,
        )
    }

    @Test
    fun `4次元最大値_axis=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.max(x = x, xi = 2, xj = 2, xk = 2, xl = 3, axis = 3)

        assertEquals(
            expected = DataBuffer.create(2f, 5f, 8f, 11f, 14f, 17f, 20f, 23f),
            actual = actual,
        )
    }
}
