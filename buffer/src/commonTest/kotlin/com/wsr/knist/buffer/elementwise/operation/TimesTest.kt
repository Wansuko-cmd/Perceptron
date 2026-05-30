@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.buffer.elementwise.operation

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.buffer.assertContentEquals
import com.wsr.knist.buffer.bufferTestRule
import kotlin.test.Test

class TimesTest {
    @Test
    fun `スカラー×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = 1f, y = x)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f),
            actual = actual,
        )
    }

    @Test
    fun `1次元×スカラー`() = bufferTestRule {
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = y, y = 1f)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f),
            actual = actual,
        )
    }

    @Test
    fun `1次元×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = x, y = y)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 9f, 16f, 25f, 36f, 49f, 64f, 81f, 100f, 121f),
            actual = actual,
        )
    }

    @Test
    fun `1次元×2次元_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(3) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = x, y = y, yi = 3, yj = 4, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 0f, 0f, 4f, 5f, 6f, 7f, 16f, 18f, 20f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `1次元×2次元_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(4) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = x, y = y, yi = 3, yj = 4, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 9f, 0f, 5f, 12f, 21f, 0f, 9f, 20f, 33f),
            actual = actual,
        )
    }

    @Test
    fun `1次元×3次元_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(2) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = x, y = y, yi = 2, yj = 2, yk = 3, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 0f, 0f, 0f, 0f, 6f, 7f, 8f, 9f, 10f, 11f),
            actual = actual,
        )
    }

    @Test
    fun `1次元×3次元_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(2) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = x, y = y, yi = 2, yj = 2, yk = 3, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 0f, 3f, 4f, 5f, 0f, 0f, 0f, 9f, 10f, 11f),
            actual = actual,
        )
    }

    @Test
    fun `1次元×3次元_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(3) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = x, y = y, yi = 2, yj = 2, yk = 3, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 0f, 4f, 10f, 0f, 7f, 16f, 0f, 10f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `2次元_axis=0×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(3) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 3, xj = 4, y = y, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 0f, 0f, 4f, 5f, 6f, 7f, 16f, 18f, 20f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `2次元_axis=1×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 3, xj = 4, y = y, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 9f, 0f, 5f, 12f, 21f, 0f, 9f, 20f, 33f),
            actual = actual,
        )
    }

    @Test
    fun `2次元×3次元_axis1=0_axis2=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(6) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 3, y = y, yi = 2, yj = 3, yk = 2, axis1 = 0, axis2 = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 2f, 3f, 8f, 10f, 18f, 21f, 32f, 36f, 50f, 55f),
            actual = actual,
        )
    }

    @Test
    fun `2次元×3次元_axis1=0_axis2=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(6) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 3, y = y, yi = 2, yj = 2, yk = 3, axis1 = 0, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 0f, 4f, 10f, 18f, 28f, 40f, 27f, 40f, 55f),
            actual = actual,
        )
    }

    @Test
    fun `2次元×3次元_axis1=1_axis2=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(6) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(12) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 3, y = y, yi = 2, yj = 2, yk = 3, axis1 = 1, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 9f, 16f, 25f, 0f, 7f, 16f, 27f, 40f, 55f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis=0×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 2, xk = 3, y = y, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 0f, 0f, 0f, 0f, 6f, 7f, 8f, 9f, 10f, 11f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis=1×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 2, xk = 3, y = y, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 0f, 3f, 4f, 5f, 0f, 0f, 0f, 9f, 10f, 11f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis=2×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(3) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 2, xk = 3, y = y, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 0f, 4f, 10f, 0f, 7f, 16f, 0f, 10f, 22f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis1=0_axis2=1×2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(6) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 3, xk = 2, y = y, yi = 2, yj = 3, axis1 = 0, axis2 = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 2f, 3f, 8f, 10f, 18f, 21f, 32f, 36f, 50f, 55f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis1=0_axis2=2×2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(6) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 2, xk = 3, y = y, yi = 2, yj = 3, axis1 = 0, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 0f, 4f, 10f, 18f, 28f, 40f, 27f, 40f, 55f),
            actual = actual,
        )
    }

    @Test
    fun `3次元_axis1=1_axis2=2×2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(12) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(6) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 2, xk = 3, y = y, yi = 2, yj = 3, axis1 = 1, axis2 = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 9f, 16f, 25f, 0f, 7f, 16f, 27f, 40f, 55f),
            actual = actual,
        )
    }

    @Test
    fun `3次元×4次元_axis1=0_axis2=1_axis3=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(16) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 0, axis2 = 1, axis3 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 2f, 3f, 8f, 10f, 18f, 21f, 32f, 36f, 50f, 55f, 72f, 78f, 98f, 105f),
            actual = actual,
        )
    }

    @Test
    fun `3次元×4次元_axis1=0_axis2=1_axis3=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(16) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 0, axis2 = 1, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 0f, 3f, 8f, 15f, 12f, 21f, 32f, 45f, 40f, 55f, 72f, 91f, 84f, 105f),
            actual = actual,
        )
    }

    @Test
    fun `3次元×4次元_axis1=0_axis2=2_axis3=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(16) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 0, axis2 = 2, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 9f, 0f, 5f, 12f, 21f, 32f, 45f, 60f, 77f, 48f, 65f, 84f, 105f),
            actual = actual,
        )
    }

    @Test
    fun `3次元×4次元_axis1=1_axis2=2_axis3=3`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(8) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(16) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2,
            y = y, yi = 2, yj = 2, yk = 2, yl = 2,
            axis1 = 1, axis2 = 2, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 9f, 16f, 25f, 36f, 49f, 0f, 9f, 20f, 33f, 48f, 65f, 84f, 105f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=0×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 0)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=1×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 1)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 0f, 0f, 4f, 5f, 6f, 7f, 0f, 0f, 0f, 0f, 12f, 13f, 14f, 15f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=2×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 2)

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 2f, 3f, 0f, 0f, 6f, 7f, 0f, 0f, 10f, 11f, 0f, 0f, 14f, 15f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis=3×1次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(2) { it.toFloat() })

        val actual = Backend.times(x = x, xi = 2, xj = 2, xk = 2, xl = 2, y = y, axis = 3)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 0f, 3f, 0f, 5f, 0f, 7f, 0f, 9f, 0f, 11f, 0f, 13f, 0f, 15f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=1×2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 0, axis2 = 1,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 0f, 0f, 4f, 5f, 6f, 7f, 16f, 18f, 20f, 22f, 36f, 39f, 42f, 45f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=2×2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 0, axis2 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 2f, 3f, 0f, 0f, 6f, 7f, 16f, 18f, 30f, 33f, 24f, 26f, 42f, 45f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=3×2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 0, axis2 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 0f, 3f, 0f, 5f, 0f, 7f, 16f, 27f, 20f, 33f, 24f, 39f, 28f, 45f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=1_axis2=2×2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 1, axis2 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 2f, 3f, 8f, 10f, 18f, 21f, 0f, 0f, 10f, 11f, 24f, 26f, 42f, 45f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=1_axis2=3×2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 1, axis2 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 0f, 3f, 8f, 15f, 12f, 21f, 0f, 9f, 0f, 11f, 24f, 39f, 28f, 45f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=2_axis2=3×2次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(4) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2,
            axis1 = 2, axis2 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 9f, 0f, 5f, 12f, 21f, 0f, 9f, 20f, 33f, 0f, 13f, 28f, 45f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=1_axis2=2×3次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(8) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2, yk = 2,
            axis1 = 0, axis2 = 1, axis3 = 2,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 0f, 2f, 3f, 8f, 10f, 18f, 21f, 32f, 36f, 50f, 55f, 72f, 78f, 98f, 105f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=0_axis2=1_axis2=3×3次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(8) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2, yk = 2,
            axis1 = 0, axis2 = 1, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 0f, 3f, 8f, 15f, 12f, 21f, 32f, 45f, 40f, 55f, 72f, 91f, 84f, 105f),
            actual = actual,
        )
    }

    @Test
    fun `4次元_axis1=1_axis2=2_axis2=3×3次元`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(16) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(8) { it.toFloat() })

        val actual = Backend.times(
            x = x, xi = 2, xj = 2, xk = 2, xl = 2,
            y = y, yi = 2, yj = 2, yk = 2,
            axis1 = 1, axis2 = 2, axis3 = 3,
        )

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 4f, 9f, 16f, 25f, 36f, 49f, 0f, 9f, 20f, 33f, 48f, 65f, 84f, 105f),
            actual = actual,
        )
    }
}
