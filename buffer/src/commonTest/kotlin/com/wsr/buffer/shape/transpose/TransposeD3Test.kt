@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.shape.transpose

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import kotlin.test.Test

class TransposeD3Test {
    val input = DataBuffer.create(FloatArray(24) { it.toFloat() })

    @Test
    fun `transpose_012=3次元転置`() = bufferTestRule {
        val result = Backend.transpose(x = input, xi = 2, xj = 3, xk = 4, axisI = 0, axisJ = 1, axisK = 2)

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 3f,
                4f, 5f, 6f, 7f,
                8f, 9f, 10f, 11f,

                12f, 13f, 14f, 15f,
                16f, 17f, 18f, 19f,
                20f, 21f, 22f, 23f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_021=3次元転置`() = bufferTestRule {
        val result = Backend.transpose(x = input, xi = 2, xj = 3, xk = 4, axisI = 0, axisJ = 2, axisK = 1)

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 4f, 8f,
                1f, 5f, 9f,
                2f, 6f, 10f,
                3f, 7f, 11f,

                12f, 16f, 20f,
                13f, 17f, 21f,
                14f, 18f, 22f,
                15f, 19f, 23f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_102=3次元転置`() = bufferTestRule {
        val result = Backend.transpose(x = input, xi = 2, xj = 3, xk = 4, axisI = 1, axisJ = 0, axisK = 2)

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 3f,
                12f, 13f, 14f, 15f,

                4f, 5f, 6f, 7f,
                16f, 17f, 18f, 19f,

                8f, 9f, 10f, 11f,
                20f, 21f, 22f, 23f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_120=3次元転置`() = bufferTestRule {
        val result = Backend.transpose(x = input, xi = 2, xj = 3, xk = 4, axisI = 1, axisJ = 2, axisK = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 12f,
                1f, 13f,
                2f, 14f,
                3f, 15f,

                4f, 16f,
                5f, 17f,
                6f, 18f,
                7f, 19f,

                8f, 20f,
                9f, 21f,
                10f, 22f,
                11f, 23f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_201=3次元転置`() = bufferTestRule {
        val result = Backend.transpose(x = input, xi = 2, xj = 3, xk = 4, axisI = 2, axisJ = 0, axisK = 1)

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 4f, 8f,
                12f, 16f, 20f,

                1f, 5f, 9f,
                13f, 17f, 21f,

                2f, 6f, 10f,
                14f, 18f, 22f,

                3f, 7f, 11f,
                15f, 19f, 23f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_210=3次元転置`() = bufferTestRule {
        val result = Backend.transpose(x = input, xi = 2, xj = 3, xk = 4, axisI = 2, axisJ = 1, axisK = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 12f,
                4f, 16f,
                8f, 20f,

                1f, 13f,
                5f, 17f,
                9f, 21f,

                2f, 14f,
                6f, 18f,
                10f, 22f,

                3f, 15f,
                7f, 19f,
                11f, 23f,
            ),
            actual = result,
        )
    }
}
