@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.buffer.shape.flip

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.buffer.assertContentEquals
import com.wsr.knist.buffer.bufferTestRule
import kotlin.test.Test

class FlipD4Test {
    val input = DataBuffer.create(FloatArray(24) { it.toFloat() })

    @Test
    fun `flip_axis0=4次元flip`() = bufferTestRule {
        val result = Backend.flip(x = input, xi = 2, xj = 3, xk = 2, xl = 2, axis = 0)

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
    fun `flip_axis1=4次元flip`() = bufferTestRule {
        val result = Backend.flip(x = input, xi = 2, xj = 3, xk = 2, xl = 2, axis = 1)

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
    fun `flip_axis2=4次元flip`() = bufferTestRule {
        val result = Backend.flip(x = input, xi = 2, xj = 3, xk = 2, xl = 2, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(
                2f, 3f, 0f, 1f,
                6f, 7f, 4f, 5f,
                10f, 11f, 8f, 9f,

                14f, 15f, 12f, 13f,
                18f, 19f, 16f, 17f,
                22f, 23f, 20f, 21f,
            ),
            actual = result,
        )
    }

    @Test
    fun `flip_axis3=4次元flip`() = bufferTestRule {
        val result = Backend.flip(x = input, xi = 2, xj = 3, xk = 2, xl = 2, axis = 3)

        assertContentEquals(
            expected = DataBuffer.create(
                1f, 0f, 3f, 2f,
                5f, 4f, 7f, 6f,
                9f, 8f, 11f, 10f,

                13f, 12f, 15f, 14f,
                17f, 16f, 19f, 18f,
                21f, 20f, 23f, 22f,
            ),
            actual = result,
        )
    }
}
