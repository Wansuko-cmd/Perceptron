@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.collection

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test
import kotlin.test.assertEquals

class AverageTest {
    @Test
    fun `1次元平均`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.average(x = x)

        assertEquals(expected = 11.5f, actual = actual)
    }

    @Test
    fun `2次元平均_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.average(x = x, xi = 4, xj = 6, axis = 0)

        assertEquals(expected = DataBuffer.create(9f, 10f, 11f, 12f, 13f, 14f), actual = actual)
    }

    @Test
    fun `2次元平均_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.average(x = x, xi = 4, xj = 6, axis = 1)

        assertEquals(expected = DataBuffer.create(2.5f, 8.5f, 14.5f, 20.5f), actual = actual)
    }

    @Test
    fun `3次元平均_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.average(x = x, xi = 4, xj = 3, xk = 2, axis = 0)

        assertEquals(expected = DataBuffer.create(9f, 10f, 11f, 12f, 13f, 14f), actual = actual)
    }

    @Test
    fun `3次元平均_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.average(x = x, xi = 4, xj = 3, xk = 2, axis = 1)

        assertEquals(expected = DataBuffer.create(2f, 3f, 8f, 9f, 14f, 15f, 20f, 21f), actual = actual)
    }

    @Test
    fun `3次元平均_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.average(x = x, xi = 4, xj = 3, xk = 2, axis = 2)

        assertEquals(
            expected = DataBuffer.create(0.5f, 2.5f, 4.5f, 6.5f, 8.5f, 10.5f, 12.5f, 14.5f, 16.5f, 18.5f, 20.5f, 22.5f),
            actual = actual,
        )
    }
}
