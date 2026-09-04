@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.buffer.linalg

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.buffer.assertContentEquals
import com.wsr.knist.buffer.bufferTestRule
import kotlin.test.Test

class MatMulTest {
    @Test
    fun `inner=内積`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(20) { it * 2f })
        val y = DataBuffer.create(FloatArray(20) { -it + 3f })

        val actual = Backend.inner(x = x, y = y, b = 4)

        assertContentEquals(expected = DataBuffer.create(0f, -300f, -1100f, -2400f), actual = actual)
    }

    @Test
    fun `D2 matMul D3=行列積`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(5) { it * 2f })
        val y = DataBuffer.create(FloatArray(20) { -it + 3f })

        val actual = Backend.matMul(x = x, y = y, transY = false, n = 4, k = 5)

        assertContentEquals(expected = DataBuffer.create(-180f, -200f, -220f, -240f), actual = actual)
    }

    @Test
    fun `D2 matMul D3T=行列積`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(5) { it * 2f })
        val y = DataBuffer.create(FloatArray(20) { -it + 3f })

        val actual = Backend.matMul(x = x, y = y, transY = true, n = 4, k = 5)

        assertContentEquals(expected = DataBuffer.create(0f, -100f, -200f, -300f), actual = actual)
    }

    @Test
    fun `D3 matMul D2=行列積`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(20) { it * 2f })
        val y = DataBuffer.create(FloatArray(5) { -it + 3f })

        val actual = Backend.matMul(x = x, transX = false, y = y, m = 4, k = 5)

        assertContentEquals(expected = DataBuffer.create(0f, 50f, 100f, 150f), actual = actual)
    }

    @Test
    fun `D3T matMul D2=行列積`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(20) { it * 2f })
        val y = DataBuffer.create(FloatArray(5) { -it + 3f })

        val actual = Backend.matMul(x = x, transX = true, y = y, m = 4, k = 5)

        assertContentEquals(expected = DataBuffer.create(0f, 10f, 20f, 30f), actual = actual)
    }

    @Test
    fun `D3 matMul D3=行列積`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(30) { it * 2f })
        val y = DataBuffer.create(FloatArray(30) { -it + 3f })

        val actual = Backend.matMul(x = x, transX = false, y = y, transY = false, m = 3, n = 3, k = 5, b = 2)

        assertContentEquals(
            expected = DataBuffer.create(
                -120f,
                -140f,
                -160f,
                -270f,
                -340f,
                -410f,
                -420f,
                -540f,
                -660f,
                -3120f,
                -3290f,
                -3460f,
                -4020f,
                -4240f,
                -4460f,
                -4920f,
                -5190f,
                -5460f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `D3 matMul D3T=行列積`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(30) { it * 2f })
        val y = DataBuffer.create(FloatArray(30) { -it + 3f })

        val actual = Backend.matMul(x = x, transX = false, y = y, transY = true, m = 3, n = 3, k = 5, b = 2)

        assertContentEquals(
            expected = DataBuffer.create(
                0f,
                -100f,
                -200f,
                50f,
                -300f,
                -650f,
                100f,
                -500f,
                -1100f,
                -2400f,
                -3250f,
                -4100f,
                -3100f,
                -4200f,
                -5300f,
                -3800f,
                -5150f,
                -6500f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `D3T matMul D3=行列積`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(30) { it * 2f })
        val y = DataBuffer.create(FloatArray(30) { -it + 3f })

        val actual = Backend.matMul(x = x, transX = true, y = y, transY = false, m = 3, n = 3, k = 5, b = 2)

        assertContentEquals(
            expected = DataBuffer.create(
                -360f,
                -420f,
                -480f,
                -390f,
                -460f,
                -530f,
                -420f,
                -500f,
                -580f,
                -3960f,
                -4170f,
                -4380f,
                -4140f,
                -4360f,
                -4580f,
                -4320f,
                -4550f,
                -4780f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `D3T matMul D3T=行列積`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(30) { it * 2f })
        val y = DataBuffer.create(FloatArray(30) { -it + 3f })

        val actual = Backend.matMul(x = x, transX = true, y = y, transY = true, m = 3, n = 3, k = 5, b = 2)

        assertContentEquals(
            expected = DataBuffer.create(
                0f,
                -300f,
                -600f,
                10f,
                -340f,
                -690f,
                20f,
                -380f,
                -780f,
                -3000f,
                -4050f,
                -5100f,
                -3140f,
                -4240f,
                -5340f,
                -3280f,
                -4430f,
                -5580f,
            ),
            actual = actual,
        )
    }
}
