@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.flip

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import kotlin.test.Test

class FlipD3Test {
    val input = DataBuffer.create(FloatArray(24) { it.toFloat() })

    @Test
    fun `flip_axis0=3次元flip`() = bufferTestRule {
        val result = Backend.flip(x = input, xi = 2, xj = 3, xk = 4, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                12f, 13f, 14f, 15f,
                16f, 17f, 18f, 19f,
                20f, 21f, 22f, 23f,

                0f, 1f, 2f, 3f,
                4f, 5f, 6f, 7f,
                8f, 9f, 10f, 11f,
            ),
            actual = result,
        )
    }

    @Test
    fun `flip_axis1=3次元flip`() = bufferTestRule {
        val result = Backend.flip(x = input, xi = 2, xj = 3, xk = 4, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(
                8f, 9f, 10f, 11f,
                4f, 5f, 6f, 7f,
                0f, 1f, 2f, 3f,

                20f, 21f, 22f, 23f,
                16f, 17f, 18f, 19f,
                12f, 13f, 14f, 15f,
            ),
            actual = result,
        )
    }

    @Test
    fun `flip_axis2=3次元flip`() = bufferTestRule {
        val result = Backend.flip(x = input, xi = 2, xj = 3, xk = 4, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(
                3f, 2f, 1f, 0f,
                7f, 6f, 5f, 4f,
                11f, 10f, 9f, 8f,

                15f, 14f, 13f, 12f,
                19f, 18f, 17f, 16f,
                23f, 22f, 21f, 20f,
            ),
            actual = result,
        )
    }
}
