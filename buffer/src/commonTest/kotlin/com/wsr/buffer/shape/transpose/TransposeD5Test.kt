@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.shape.transpose

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import kotlin.test.Test

class TransposeD5Test {
    val input = DataBuffer.create(FloatArray(32) { it.toFloat() })

    @Test
    fun `transpose_01234=5次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 2, xk = 2, xl = 2, xm = 2,
            axisI = 0, axisJ = 1, axisK = 2, axisL = 3, axisM = 4,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f,
                8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f,
                16f, 17f, 18f, 19f, 20f, 21f, 22f, 23f,
                24f, 25f, 26f, 27f, 28f, 29f, 30f, 31f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_01243=5次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 2, xk = 2, xl = 2, xm = 2,
            axisI = 0, axisJ = 1, axisK = 2, axisL = 4, axisM = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 2f, 1f, 3f, 4f, 6f, 5f, 7f,
                8f, 10f, 9f, 11f, 12f, 14f, 13f, 15f,
                16f, 18f, 17f, 19f, 20f, 22f, 21f, 23f,
                24f, 26f, 25f, 27f, 28f, 30f, 29f, 31f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_01324=5次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 2, xk = 2, xl = 2, xm = 2,
            axisI = 0, axisJ = 1, axisK = 3, axisL = 2, axisM = 4,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 1f, 4f, 5f, 2f, 3f, 6f, 7f,
                8f, 9f, 12f, 13f, 10f, 11f, 14f, 15f,
                16f, 17f, 20f, 21f, 18f, 19f, 22f, 23f,
                24f, 25f, 28f, 29f, 26f, 27f, 30f, 31f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_02134=5次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 2, xk = 2, xl = 2, xm = 2,
            axisI = 0, axisJ = 2, axisK = 1, axisL = 3, axisM = 4,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 3f, 8f, 9f, 10f, 11f,
                4f, 5f, 6f, 7f, 12f, 13f, 14f, 15f,
                16f, 17f, 18f, 19f, 24f, 25f, 26f, 27f,
                20f, 21f, 22f, 23f, 28f, 29f, 30f, 31f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_10234=5次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 2, xk = 2, xl = 2, xm = 2,
            axisI = 1, axisJ = 0, axisK = 2, axisL = 3, axisM = 4,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f,
                16f, 17f, 18f, 19f, 20f, 21f, 22f, 23f,
                8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f,
                24f, 25f, 26f, 27f, 28f, 29f, 30f, 31f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_43210=5次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 2, xk = 2, xl = 2, xm = 2,
            axisI = 4, axisJ = 3, axisK = 2, axisL = 1, axisM = 0,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 16f, 8f, 24f, 4f, 20f, 12f, 28f,
                2f, 18f, 10f, 26f, 6f, 22f, 14f, 30f,
                1f, 17f, 9f, 25f, 5f, 21f, 13f, 29f,
                3f, 19f, 11f, 27f, 7f, 23f, 15f, 31f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_12340=5次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 2, xk = 2, xl = 2, xm = 2,
            axisI = 1, axisJ = 2, axisK = 3, axisL = 4, axisM = 0,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 16f, 1f, 17f, 2f, 18f, 3f, 19f,
                4f, 20f, 5f, 21f, 6f, 22f, 7f, 23f,
                8f, 24f, 9f, 25f, 10f, 26f, 11f, 27f,
                12f, 28f, 13f, 29f, 14f, 30f, 15f, 31f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_04321=5次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 2, xk = 2, xl = 2, xm = 2,
            axisI = 0, axisJ = 4, axisK = 3, axisL = 2, axisM = 1,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 8f, 4f, 12f, 2f, 10f, 6f, 14f,
                1f, 9f, 5f, 13f, 3f, 11f, 7f, 15f,
                16f, 24f, 20f, 28f, 18f, 26f, 22f, 30f,
                17f, 25f, 21f, 29f, 19f, 27f, 23f, 31f,
            ),
            actual = result,
        )
    }
}
