@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.operation

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test

class PlusTest {
    @Test
    fun `スカラー+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = 1f, y = x)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f),
            actual = actual,
        )
    }

    @Test
    fun `1次元+スカラー`() = bufferTestRule {
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = y, y = 1f)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f),
            actual = actual,
        )
    }

    @Test
    fun `1次元+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = x, y = y)

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f, 18f, 20f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `1次元+2次元_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(3) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = x, y = y, yi = 3, yj = 4, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 3f, 5f, 6f, 7f, 8f, 10f, 11f, 12f, 13f),
            actual = actual,
        )
    }

    @Test
    fun `1次元+2次元_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(4) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = x, y = y, yi = 3, yj = 4, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 6f, 4f, 6f, 8f, 10f, 8f, 10f, 12f, 14f),
            actual = actual,
        )
    }

    @Test
    fun `1次元+3次元_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(2) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = x, y = y, yi = 2, yj = 2, yk = 3, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 3f, 4f, 5f, 7f, 8f, 9f, 10f, 11f, 12f),
            actual = actual,
        )
    }

    @Test
    fun `1次元+3次元_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(2) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = x, y = y, yi = 2, yj = 2, yk = 3, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 4f, 5f, 6f, 6f, 7f, 8f, 10f, 11f, 12f),
            actual = actual,
        )
    }

    @Test
    fun `1次元+3次元_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(3) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = x, y = y, yi = 2, yj = 2, yk = 3, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 3f, 5f, 7f, 6f, 8f, 10f, 9f, 11f, 13f),
            actual = actual,
        )
    }

    @Test
    fun `2次元_axis=0+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(3) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 3, xj = 4, y = y, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 3f, 5f, 6f, 7f, 8f, 10f, 11f, 12f, 13f),
            actual = actual,
        )
    }

    @Test
    fun `2次元_axis=1+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 3, xj = 4, y = y, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 6f, 4f, 6f, 8f, 10f, 8f, 10f, 12f, 14f),
            actual = actual,
        )
    }

    @Test
    fun `2次元+3次元_axis1=0_axis2=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(6) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 3, y = y, yi = 2, yj = 3, yk = 2, axis1 = 0, axis2 = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 3f, 4f, 6f, 7f, 9f, 10f, 12f, 13f, 15f, 16f),
            actual = actual,
        )
    }

    @Test
    fun `2次元+3次元_axis1=0_axis2=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(6) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 3, y = y, yi = 2, yj = 2, yk = 3, axis1 = 0, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 3f, 5f, 7f, 9f, 11f, 13f, 12f, 14f, 16f),
            actual = actual,
        )
    }

    @Test
    fun `2次元+3次元_axis1=1_axis2=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(6) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 3, y = y, yi = 2, yj = 2, yk = 3, axis1 = 1, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 6f, 8f, 10f, 6f, 8f, 10f, 12f, 14f, 16f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis=0+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 2, xk = 3, y = y, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 3f, 4f, 5f, 7f, 8f, 9f, 10f, 11f, 12f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis=1+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 2, xk = 3, y = y, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 4f, 5f, 6f, 6f, 7f, 8f, 10f, 11f, 12f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis=2+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(3) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 2, xk = 3, y = y, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 3f, 5f, 7f, 6f, 8f, 10f, 9f, 11f, 13f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis1=0_axis2=1+2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(6) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 3, xk = 2, y = y, yi = 2, yj = 3, axis1 = 0, axis2 = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 3f, 4f, 6f, 7f, 9f, 10f, 12f, 13f, 15f, 16f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis1=0_axis2=2+2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(6) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 2, xk = 3, y = y, yi = 2, yj = 3, axis1 = 0, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 3f, 5f, 7f, 9f, 11f, 13f, 12f, 14f, 16f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis1=1_axis2=2+2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(6) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 2, xk = 3, y = y, yi = 2, yj = 3, axis1 = 1, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 6f, 8f, 10f, 6f, 8f, 10f, 12f, 14f, 16f),
            actual = actual,
        )
    }

    @Test
    fun `3次元+4次元_axis1=0_axis2=1_axis3=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(16) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 0, axis2 = 1, axis3 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 3f, 4f, 6f, 7f, 9f, 10f, 12f, 13f, 15f, 16f, 18f, 19f, 21f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `3次元+4次元_axis1=0_axis2=1_axis3=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(16) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 0, axis2 = 1, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 2f, 4f, 6f, 8f, 8f, 10f, 12f, 14f, 14f, 16f, 18f, 20f, 20f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `3次元+4次元_axis1=0_axis2=2_axis3=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(16) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 0, axis2 = 2, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 6f, 4f, 6f, 8f, 10f, 12f, 14f, 16f, 18f, 16f, 18f, 20f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `3次元+4次元_axis1=1_axis2=2_axis3=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(16) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 1, axis2 = 2, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 6f, 8f, 10f, 12f, 14f, 8f, 10f, 12f, 14f, 16f, 18f, 20f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=0+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=1+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 3f, 5f, 6f, 7f, 8f, 8f, 9f, 10f, 11f, 13f, 14f, 15f, 16f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=2+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 3f, 4f, 4f, 5f, 7f, 8f, 8f, 9f, 11f, 12f, 12f, 13f, 15f, 16f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=3+1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.plus(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 3)

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 2f, 4f, 4f, 6f, 6f, 8f, 8f, 10f, 10f, 12f, 12f, 14f, 14f, 16f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=1+2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 0, axis2 = 1,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 3f, 5f, 6f, 7f, 8f, 10f, 11f, 12f, 13f, 15f, 16f, 17f, 18f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=2+2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 0, axis2 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 3f, 4f, 4f, 5f, 7f, 8f, 10f, 11f, 13f, 14f, 14f, 15f, 17f, 18f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=3+2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 0, axis2 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 2f, 4f, 4f, 6f, 6f, 8f, 10f, 12f, 12f, 14f, 14f, 16f, 16f, 18f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=1_axis2=2+2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 1, axis2 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 3f, 4f, 6f, 7f, 9f, 10f, 8f, 9f, 11f, 12f, 14f, 15f, 17f, 18f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=1_axis2=3+2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 1, axis2 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 2f, 4f, 6f, 8f, 8f, 10f, 8f, 10f, 10f, 12f, 14f, 16f, 16f, 18f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=2_axis2=3+2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 2, axis2 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 6f, 4f, 6f, 8f, 10f, 8f, 10f, 12f, 14f, 12f, 14f, 16f, 18f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=1_axis2=2+3次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(8) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2, yk = 2,
            axis1 = 0, axis2 = 1, axis3 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 3f, 4f, 6f, 7f, 9f, 10f, 12f, 13f, 15f, 16f, 18f, 19f, 21f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=1_axis2=3+3次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(8) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2, yk = 2,
            axis1 = 0, axis2 = 1, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 2f, 4f, 6f, 8f, 8f, 10f, 12f, 14f, 14f, 16f, 18f, 20f, 20f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=1_axis2=2_axis2=3+3次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(8) { it.toFloat() })

        val actual = Backend.plus(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2, yk = 2,
            axis1 = 1, axis2 = 2, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 2f, 4f, 6f, 8f, 10f, 12f, 14f, 8f, 10f, 12f, 14f, 16f, 18f, 20f, 22f),
            actual = actual,
        )
    }
}
