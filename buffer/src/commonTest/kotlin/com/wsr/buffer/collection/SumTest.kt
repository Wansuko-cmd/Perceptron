@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.collection

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import kotlin.test.Test
class SumTest {
    @Test
    fun `1次元内累積値`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x)

        assertContentEquals(expected = DataBuffer.create(276f), actual = actual, absoluteTolerance = 1e-4f)
    }

    @Test
    fun `2次元累積値_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 4, xj = 6, axis = 0)

        assertContentEquals(expected = DataBuffer.create(36f, 40f, 44f, 48f, 52f, 56f), actual = actual)
    }

    @Test
    fun `2次元累積値_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 4, xj = 6, axis = 1)

        assertContentEquals(expected = DataBuffer.create(15f, 51f, 87f, 123f), actual = actual)
    }

    @Test
    fun `3次元累積値_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 4, xj = 3, xk = 2, axis = 0)

        assertContentEquals(expected = DataBuffer.create(36f, 40f, 44f, 48f, 52f, 56f), actual = actual)
    }

    @Test
    fun `3次元累積値_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 4, xj = 3, xk = 2, axis = 1)

        assertContentEquals(expected = DataBuffer.create(6f, 9f, 24f, 27f, 42f, 45f, 60f, 63f), actual = actual)
    }

    @Test
    fun `3次元累積値_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.sum(x = x, xi = 4, xj = 3, xk = 2, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(1f, 5f, 9f, 13f, 17f, 21f, 25f, 29f, 33f, 37f, 41f, 45f),
            actual = actual,
        )
    }
}
