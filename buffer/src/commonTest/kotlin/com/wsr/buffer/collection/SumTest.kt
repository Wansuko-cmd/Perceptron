@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.collection

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test
import kotlin.test.assertEquals

class SumTest {
    @Test
    fun `1次元内累積値`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x)

        assertEquals(expected = 276f, actual = actual)
    }

    @Test
    fun `2次元累積値_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 4, xj = 6, axis = 0)

        assertEquals(expected = DataBuffer.create(36f, 40f, 44f, 48f, 52f, 56f), actual = actual)
    }

    @Test
    fun `2次元累積値_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 4, xj = 6, axis = 1)

        assertEquals(expected = DataBuffer.create(15f, 51f, 87f, 123f), actual = actual)
    }

    @Test
    fun `3次元累積値_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 4, xj = 3, xk = 2, axis = 0)

        assertEquals(expected = DataBuffer.create(36f, 40f, 44f, 48f, 52f, 56f), actual = actual)
    }

    @Test
    fun `3次元累積値_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 4, xj = 3, xk = 2, axis = 1)

        assertEquals(expected = DataBuffer.create(6f, 9f, 24f, 27f, 42f, 45f, 60f, 63f), actual = actual)
    }

    @Test
    fun `3次元累積値_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 4, xj = 3, xk = 2, axis = 2)

        assertEquals(
            expected = DataBuffer.create(1f, 5f, 9f, 13f, 17f, 21f, 25f, 29f, 33f, 37f, 41f, 45f),
            actual = actual,
        )
    }

    @Test
    fun `4次元累積値_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 2, xj = 2, xk = 2, xl = 3, axis = 0)

        assertEquals(
            expected = DataBuffer.create(12f, 14f, 16f, 18f, 20f, 22f, 24f, 26f, 28f, 30f, 32f, 34f),
            actual = actual,
        )
    }

    @Test
    fun `4次元累積値_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 2, xj = 2, xk = 2, xl = 3, axis = 1)

        assertEquals(
            expected = DataBuffer.create(6f, 8f, 10f, 12f, 14f, 16f, 30f, 32f, 34f, 36f, 38f, 40f),
            actual = actual,
        )
    }

    @Test
    fun `4次元累積値_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 2, xj = 2, xk = 2, xl = 3, axis = 2)

        assertEquals(
            expected = DataBuffer.create(3f, 5f, 7f, 15f, 17f, 19f, 27f, 29f, 31f, 39f, 41f, 43f),
            actual = actual,
        )
    }

    @Test
    fun `4次元累積値_axis=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 2, xj = 2, xk = 2, xl = 3, axis = 3)

        assertEquals(
            expected = DataBuffer.create(3f, 12f, 21f, 30f, 39f, 48f, 57f, 66f),
            actual = actual,
        )
    }
}
