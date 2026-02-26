@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.transpose

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertEquals
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test

class TransposeD4Test {
    val input = DataBuffer.create(FloatArray(36) { it.toFloat() })

    @Test
    fun `transpose_0123=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 0, axisJ = 1, axisK = 2, axisL = 3,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 3f, 4f, 5f,
                6f, 7f, 8f, 9f, 10f, 11f,
                12f, 13f, 14f, 15f, 16f, 17f,

                18f, 19f, 20f, 21f, 22f, 23f,
                24f, 25f, 26f, 27f, 28f, 29f,
                30f, 31f, 32f, 33f, 34f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_0132=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 0, axisJ = 1, axisK = 3, axisL = 2,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 3f, 1f, 4f, 2f, 5f,
                6f, 9f, 7f, 10f, 8f, 11f,
                12f, 15f, 13f, 16f, 14f, 17f,

                18f, 21f, 19f, 22f, 20f, 23f,
                24f, 27f, 25f, 28f, 26f, 29f,
                30f, 33f, 31f, 34f, 32f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_0213=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 0, axisJ = 2, axisK = 1, axisL = 3,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 6f, 7f, 8f, 12f, 13f, 14f,
                3f, 4f, 5f, 9f, 10f, 11f, 15f, 16f, 17f,

                18f, 19f, 20f, 24f, 25f, 26f, 30f, 31f, 32f,
                21f, 22f, 23f, 27f, 28f, 29f, 33f, 34f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_0231=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 0, axisJ = 2, axisK = 3, axisL = 1,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 6f, 12f, 1f, 7f, 13f, 2f, 8f, 14f,
                3f, 9f, 15f, 4f, 10f, 16f, 5f, 11f, 17f,

                18f, 24f, 30f, 19f, 25f, 31f, 20f, 26f, 32f,
                21f, 27f, 33f, 22f, 28f, 34f, 23f, 29f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_0312=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 0, axisJ = 3, axisK = 1, axisL = 2,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 3f, 6f, 9f, 12f, 15f,
                1f, 4f, 7f, 10f, 13f, 16f,

                2f, 5f, 8f, 11f, 14f, 17f,
                18f, 21f, 24f, 27f, 30f, 33f,

                19f, 22f, 25f, 28f, 31f, 34f,
                20f, 23f, 26f, 29f, 32f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_0321=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 0, axisJ = 3, axisK = 2, axisL = 1,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 6f, 12f, 3f, 9f, 15f,
                1f, 7f, 13f, 4f, 10f, 16f,
                2f, 8f, 14f, 5f, 11f, 17f,

                18f, 24f, 30f, 21f, 27f, 33f,
                19f, 25f, 31f, 22f, 28f, 34f,
                20f, 26f, 32f, 23f, 29f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_1023=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 1, axisJ = 0, axisK = 2, axisL = 3,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 3f, 4f, 5f,
                18f, 19f, 20f, 21f, 22f, 23f,

                6f, 7f, 8f, 9f, 10f, 11f,
                24f, 25f, 26f, 27f, 28f, 29f,

                12f, 13f, 14f, 15f, 16f, 17f,
                30f, 31f, 32f, 33f, 34f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_1032=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 1, axisJ = 0, axisK = 3, axisL = 2,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 3f, 1f, 4f, 2f, 5f,
                18f, 21f, 19f, 22f, 20f, 23f,

                6f, 9f, 7f, 10f, 8f, 11f,
                24f, 27f, 25f, 28f, 26f, 29f,

                12f, 15f, 13f, 16f, 14f, 17f,
                30f, 33f, 31f, 34f, 32f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_1203=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 1, axisJ = 2, axisK = 0, axisL = 3,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 18f, 19f, 20f,
                3f, 4f, 5f, 21f, 22f, 23f,

                6f, 7f, 8f, 24f, 25f, 26f,
                9f, 10f, 11f, 27f, 28f, 29f,

                12f, 13f, 14f, 30f, 31f, 32f,
                15f, 16f, 17f, 33f, 34f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_1230=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 1, axisJ = 2, axisK = 3, axisL = 0,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 18f, 1f, 19f, 2f, 20f,
                3f, 21f, 4f, 22f, 5f, 23f,

                6f, 24f, 7f, 25f, 8f, 26f,
                9f, 27f, 10f, 28f, 11f, 29f,

                12f, 30f, 13f, 31f, 14f, 32f,
                15f, 33f, 16f, 34f, 17f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_1302=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 1, axisJ = 3, axisK = 0, axisL = 2,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 3f, 18f, 21f,
                1f, 4f, 19f, 22f,
                2f, 5f, 20f, 23f,

                6f, 9f, 24f, 27f,
                7f, 10f, 25f, 28f,
                8f, 11f, 26f, 29f,

                12f, 15f, 30f, 33f,
                13f, 16f, 31f, 34f,
                14f, 17f, 32f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_1320=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 1, axisJ = 3, axisK = 2, axisL = 0,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 18f, 3f, 21f,
                1f, 19f, 4f, 22f,
                2f, 20f, 5f, 23f,

                6f, 24f, 9f, 27f,
                7f, 25f, 10f, 28f,
                8f, 26f, 11f, 29f,

                12f, 30f, 15f, 33f,
                13f, 31f, 16f, 34f,
                14f, 32f, 17f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_2013=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 2, axisJ = 0, axisK = 1, axisL = 3,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 6f, 7f, 8f, 12f, 13f, 14f,
                18f, 19f, 20f, 24f, 25f, 26f, 30f, 31f, 32f,

                3f, 4f, 5f, 9f, 10f, 11f, 15f, 16f, 17f,
                21f, 22f, 23f, 27f, 28f, 29f, 33f, 34f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_2031=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 2, axisJ = 0, axisK = 3, axisL = 1,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 6f, 12f, 1f, 7f, 13f, 2f, 8f, 14f,
                18f, 24f, 30f, 19f, 25f, 31f, 20f, 26f, 32f,

                3f, 9f, 15f, 4f, 10f, 16f, 5f, 11f, 17f,
                21f, 27f, 33f, 22f, 28f, 34f, 23f, 29f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_2103=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 2, axisJ = 1, axisK = 0, axisL = 3,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f, 18f, 19f, 20f,
                6f, 7f, 8f, 24f, 25f, 26f,
                12f, 13f, 14f, 30f, 31f, 32f,

                3f, 4f, 5f, 21f, 22f, 23f,
                9f, 10f, 11f, 27f, 28f, 29f,
                15f, 16f, 17f, 33f, 34f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_2130=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 2, axisJ = 1, axisK = 3, axisL = 0,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 18f, 1f, 19f, 2f, 20f,
                6f, 24f, 7f, 25f, 8f, 26f,
                12f, 30f, 13f, 31f, 14f, 32f,

                3f, 21f, 4f, 22f, 5f, 23f,
                9f, 27f, 10f, 28f, 11f, 29f,
                15f, 33f, 16f, 34f, 17f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_2301=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 2, axisJ = 3, axisK = 0, axisL = 1,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 6f, 12f, 18f, 24f, 30f,
                1f, 7f, 13f, 19f, 25f, 31f,
                2f, 8f, 14f, 20f, 26f, 32f,

                3f, 9f, 15f, 21f, 27f, 33f,
                4f, 10f, 16f, 22f, 28f, 34f,
                5f, 11f, 17f, 23f, 29f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_2310=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 2, axisJ = 3, axisK = 1, axisL = 0,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 18f, 6f, 24f, 12f, 30f,
                1f, 19f, 7f, 25f, 13f, 31f,
                2f, 20f, 8f, 26f, 14f, 32f,

                3f, 21f, 9f, 27f, 15f, 33f,
                4f, 22f, 10f, 28f, 16f, 34f,
                5f, 23f, 11f, 29f, 17f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_3012=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 3, axisJ = 0, axisK = 1, axisL = 2,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 3f, 6f, 9f, 12f, 15f,
                18f, 21f, 24f, 27f, 30f, 33f,

                1f, 4f, 7f, 10f, 13f, 16f,
                19f, 22f, 25f, 28f, 31f, 34f,

                2f, 5f, 8f, 11f, 14f, 17f,
                20f, 23f, 26f, 29f, 32f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_3021=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 3, axisJ = 0, axisK = 2, axisL = 1,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 6f, 12f, 3f, 9f, 15f,
                18f, 24f, 30f, 21f, 27f, 33f,

                1f, 7f, 13f, 4f, 10f, 16f,
                19f, 25f, 31f, 22f, 28f, 34f,

                2f, 8f, 14f, 5f, 11f, 17f,
                20f, 26f, 32f, 23f, 29f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_3102=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 3, axisJ = 1, axisK = 0, axisL = 2,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 3f, 18f, 21f,
                6f, 9f, 24f, 27f,
                12f, 15f, 30f, 33f,

                1f, 4f, 19f, 22f,
                7f, 10f, 25f, 28f,
                13f, 16f, 31f, 34f,

                2f, 5f, 20f, 23f,
                8f, 11f, 26f, 29f,
                14f, 17f, 32f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_3120=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 3, axisJ = 1, axisK = 2, axisL = 0,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 18f, 3f, 21f,
                6f, 24f, 9f, 27f,
                12f, 30f, 15f, 33f,

                1f, 19f, 4f, 22f,
                7f, 25f, 10f, 28f,
                13f, 31f, 16f, 34f,

                2f, 20f, 5f, 23f,
                8f, 26f, 11f, 29f,
                14f, 32f, 17f, 35f,
            ),
            actual = result,
        )
    }

    @Test
    fun `transpose_3210=4次元転置`() = bufferTestRule {
        val result = Backend.transpose(
            x = input, xi = 2, xj = 3, xk = 2, xl = 3,
            axisI = 3, axisJ = 2, axisK = 1, axisL = 0,
        )

        assertEquals(
            expected = DataBuffer.create(
                0f, 18f, 6f, 24f, 12f, 30f,
                3f, 21f, 9f, 27f, 15f, 33f,

                1f, 19f, 7f, 25f, 13f, 31f,
                4f, 22f, 10f, 28f, 16f, 34f,

                2f, 20f, 8f, 26f, 14f, 32f,
                5f, 23f, 11f, 29f, 17f, 35f,
            ),
            actual = result,
        )
    }
}
