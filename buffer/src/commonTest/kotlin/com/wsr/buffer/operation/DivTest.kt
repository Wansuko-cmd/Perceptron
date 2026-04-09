@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.operation

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import kotlin.test.Test

class DivTest {
    @Test
    fun `スカラー÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = 1f, y = x)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                0.5f,
                0.3333f,
                0.25f,
                0.2f,
                0.1666f,
                0.1428f,
                0.125f,
                0.1111f,
                0.1f,
                0.0909f,
                0.08333f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `1次元÷スカラー`() = bufferTestRule {
        val y = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = y, y = 1f)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f),
            actual = actual,
        )
    }

    @Test
    fun `1次元÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it + 1f })
        val y = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = x, y = y)

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f),
            actual = actual,
        )
    }

    @Test
    fun `1次元÷2次元_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(3) { it + 1f })
        val y = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = x, y = y, yi = 3, yj = 4, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                0.5f,
                0.3333f,
                0.25f,
                0.4f,
                0.3333f,
                0.2857f,
                0.25f,
                0.3333f,
                0.3f,
                0.2727f,
                0.25f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `1次元÷2次元_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(4) { it + 1f })
        val y = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = x, y = y, yi = 3, yj = 4, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 1f, 1f, 0.2f, 0.3333f, 0.4285f, 0.5f, 0.1111f, 0.2f, 0.2727f, 0.3333f),
            actual = actual,
        )
    }

    @Test
    fun `1次元÷3次元_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(2) { it + 1f })
        val y = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = x, y = y, yi = 2, yj = 2, yk = 3, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                0.5f,
                0.3333f,
                0.25f,
                0.2f,
                0.1666f,
                0.2857f,
                0.25f,
                0.2222f,
                0.2f,
                0.1818f,
                0.1666f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `1次元÷3次元_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(2) { it + 1f })
        val y = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = x, y = y, yi = 2, yj = 2, yk = 3, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                0.5f,
                0.3333f,
                0.5f,
                0.4f,
                0.3333f,
                0.1428f,
                0.125f,
                0.1111f,
                0.2f,
                0.1818f,
                0.1666f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `1次元÷3次元_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(3) { it + 1f })
        val y = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = x, y = y, yi = 2, yj = 2, yk = 3, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 1f, 0.25f, 0.4f, 0.5f, 0.1428f, 0.25f, 0.3333f, 0.1f, 0.1818f, 0.25f),
            actual = actual,
        )
    }

    @Test
    fun `2次元_axis=0÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it + 1f })
        val y = DataBuffer.create(FloatArray(3) { it + 1f })

        val actual = Backend.div(x = x, xi = 3, xj = 4, y = y, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 3f, 4f, 2.5f, 3f, 3.5f, 4f, 3f, 3.3333f, 3.6666f, 4f),
            actual = actual,
        )
    }

    @Test
    fun `2次元_axis=1÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it + 1f })
        val y = DataBuffer.create(FloatArray(4) { it + 1f })

        val actual = Backend.div(x = x, xi = 3, xj = 4, y = y, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 1f, 1f, 5f, 3f, 2.3333f, 2f, 9f, 5f, 3.6666f, 3f),
            actual = actual,
        )
    }

    @Test
    fun `2次元÷3次元_axis1=0_axis2=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(6) { it + 1f })
        val y = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 3, y = y, yi = 2, yj = 3, yk = 2, axis1 = 0, axis2 = 1)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                0.5f,
                0.6666f,
                0.5f,
                0.6f,
                0.5f,
                0.5714f,
                0.5f,
                0.5555f,
                0.5f,
                0.5454f,
                0.5f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `2次元÷3次元_axis1=0_axis2=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(6) { it + 1f })
        val y = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 3, y = y, yi = 2, yj = 2, yk = 3, axis1 = 0, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 1f, 0.25f, 0.4f, 0.5f, 0.5714f, 0.625f, 0.6666f, 0.4f, 0.4545f, 0.5f),
            actual = actual,
        )
    }

    @Test
    fun `2次元÷3次元_axis1=1_axis2=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(6) { it + 1f })
        val y = DataBuffer.create(FloatArray(12) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 3, y = y, yi = 2, yj = 2, yk = 3, axis1 = 1, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 1f, 1f, 1f, 1f, 0.1428f, 0.25f, 0.3333f, 0.4f, 0.4545f, 0.5f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis=0÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it + 1f })
        val y = DataBuffer.create(FloatArray(2) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 2, xk = 3, y = y, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 3f, 4f, 5f, 6f, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis=1÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it + 1f })
        val y = DataBuffer.create(FloatArray(2) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 2, xk = 3, y = y, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 3f, 2f, 2.5f, 3f, 7f, 8f, 9f, 5f, 5.5f, 6f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis=2÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it + 1f })
        val y = DataBuffer.create(FloatArray(3) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 2, xk = 3, y = y, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 1f, 4f, 2.5f, 2f, 7f, 4f, 3f, 10f, 5.5f, 4f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis1=0_axis2=1÷2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it + 1f })
        val y = DataBuffer.create(FloatArray(6) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 3, xk = 2, y = y, yi = 2, yj = 3, axis1 = 0, axis2 = 1)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 1.5f, 2f, 1.6666f, 2f, 1.75f, 2f, 1.8f, 2f, 1.8333f, 2f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis1=0_axis2=2÷2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it + 1f })
        val y = DataBuffer.create(FloatArray(6) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 2, xk = 3, y = y, yi = 2, yj = 3, axis1 = 0, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 1f, 4f, 2.5f, 2f, 1.75f, 1.6f, 1.5f, 2.5f, 2.2f, 2f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis1=1_axis2=2÷2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it + 1f })
        val y = DataBuffer.create(FloatArray(6) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 2, xk = 3, y = y, yi = 2, yj = 3, axis1 = 1, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 1f, 1f, 1f, 1f, 7f, 4f, 3f, 2.5f, 2.2f, 2f),
            actual = actual,
        )
    }

    @Test
    fun `3次元÷4次元_axis1=0_axis2=1_axis3=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it + 1f })
        val y = DataBuffer.create(FloatArray(16) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 0, axis2 = 1, axis3 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                0.5f,
                0.6666f,
                0.5f,
                0.6f,
                0.5f,
                0.5714f,
                0.5f,
                0.5555f,
                0.5f,
                0.5454f,
                0.5f,
                0.5384f,
                0.5f,
                0.5333f,
                0.5f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `3次元÷4次元_axis1=0_axis2=1_axis3=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it + 1f })
        val y = DataBuffer.create(FloatArray(16) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 0, axis2 = 1, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                1f,
                0.3333f,
                0.5f,
                0.6f,
                0.6666f,
                0.4285f,
                0.5f,
                0.5555f,
                0.6f,
                0.4545f,
                0.5f,
                0.5384f,
                0.5714f,
                0.4666f,
                0.5f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `3次元÷4次元_axis1=0_axis2=2_axis3=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it + 1f })
        val y = DataBuffer.create(FloatArray(16) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 0, axis2 = 2, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                1f,
                1f,
                1f,
                0.2f,
                0.3333f,
                0.4285f,
                0.5f,
                0.5555f,
                0.6f,
                0.6363f,
                0.6666f,
                0.3846f,
                0.4285f,
                0.4666f,
                0.5f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `3次元÷4次元_axis1=1_axis2=2_axis3=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it + 1f })
        val y = DataBuffer.create(FloatArray(16) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 1, axis2 = 2, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                1f,
                1f,
                1f,
                1f,
                1f,
                1f,
                1f,
                0.1111f,
                0.2f,
                0.2727f,
                0.3333f,
                0.3846f,
                0.4285f,
                0.4666f,
                0.5f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=0÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(2) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 7.5f, 8f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=1÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(2) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 3f, 4f, 2.5f, 3f, 3.5f, 4f, 9f, 10f, 11f, 12f, 6.5f, 7f, 7.5f, 8f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=2÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(2) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 1.5f, 2f, 5f, 6f, 3.5f, 4f, 9f, 10f, 5.5f, 6f, 13f, 14f, 7.5f, 8f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=3÷1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(2) { it + 1f })

        val actual = Backend.div(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 3)

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 3f, 2f, 5f, 3f, 7f, 4f, 9f, 5f, 11f, 6f, 13f, 7f, 15f, 8f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=1÷2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(4) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 0, axis2 = 1,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                2f,
                3f,
                4f,
                2.5f,
                3f,
                3.5f,
                4f,
                3f,
                3.3333f,
                3.6666f,
                4f,
                3.25f,
                3.5f,
                3.75f,
                4f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=2÷2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(4) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 0, axis2 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                2f,
                1.5f,
                2f,
                5f,
                6f,
                3.5f,
                4f,
                3f,
                3.3333f,
                2.75f,
                3f,
                4.3333f,
                4.6666f,
                3.75f,
                4f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=3÷2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(4) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 0, axis2 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 3f, 2f, 5f, 3f, 7f, 4f, 3f, 2.5f, 3.6666f, 3f, 4.3333f, 3.5f, 5f, 4f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=1_axis2=2÷2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(4) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 1, axis2 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                2f,
                1.5f,
                2f,
                1.6666f,
                2f,
                1.75f,
                2f,
                9f,
                10f,
                5.5f,
                6f,
                4.3333f,
                4.6666f,
                3.75f,
                4f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=1_axis2=3÷2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(4) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 1, axis2 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                1f,
                3f,
                2f,
                1.6666f,
                1.5f,
                2.3333f,
                2f,
                9f,
                5f,
                11f,
                6f,
                4.3333f,
                3.5f,
                5f,
                4f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=2_axis2=3÷2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(4) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 2, axis2 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(1f, 1f, 1f, 1f, 5f, 3f, 2.3333f, 2f, 9f, 5f, 3.6666f, 3f, 13f, 7f, 5f, 4f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=1_axis2=2÷3次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(8) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2, yk = 2,
            axis1 = 0, axis2 = 1, axis3 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                2f,
                1.5f,
                2f,
                1.6666f,
                2f,
                1.75f,
                2f,
                1.8f,
                2f,
                1.8333f,
                2f,
                1.8571f,
                2f,
                1.875f,
                2f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=1_axis2=3÷3次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(8) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2, yk = 2,
            axis1 = 0, axis2 = 1, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                1f,
                3f,
                2f,
                1.6666f,
                1.5f,
                2.3333f,
                2f,
                1.8f,
                1.6666f,
                2.2f,
                2f,
                1.8571f,
                1.75f,
                2.1428f,
                2f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=1_axis2=2_axis2=3÷3次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it + 1f })
        val y = DataBuffer.create(FloatArray(8) { it + 1f })

        val actual = Backend.div(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2, yk = 2,
            axis1 = 1, axis2 = 2, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                1f,
                1f,
                1f,
                1f,
                1f,
                1f,
                1f,
                9f,
                5f,
                3.6666f,
                3f,
                2.6f,
                2.3333f,
                2.1428f,
                2f,
            ),
            actual = actual,
        )
    }
}
