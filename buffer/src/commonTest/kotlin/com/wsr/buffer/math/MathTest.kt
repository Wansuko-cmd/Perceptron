@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.math

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test

class MathTest {
    val input = DataBuffer.create(FloatArray(10) { it.toFloat() })

    @Test
    fun `exp=指数関数`() = bufferTestRule {
        val actual = Backend.exp(x = input)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                2.7182f,
                7.3890f,
                20.0855f,
                54.5981f,
                148.4131f,
                403.4288f,
                1096.6332f,
                2980.958f,
                8103.084f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `ln=自然対数`() = bufferTestRule {
        val actual = Backend.ln(x = input, e = 1e-5f)

        assertContentEquals(
            expected = DataBuffer.create(
                -11.5129f,
                0.0000f,
                0.6931f,
                1.0986f,
                1.3862f,
                1.6094f,
                1.7917f,
                1.9459f,
                2.0794f,
                2.1972f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `pow=階乗`() = bufferTestRule {
        val actual = Backend.pow(x = input, n = 3)

        assertContentEquals(
            expected = DataBuffer.create(0f, 1f, 8f, 27f, 64f, 125f, 216f, 343f, 512f, 729f),
            actual = actual,
        )
    }

    @Test
    fun `sqrt=平方根`() = bufferTestRule {
        val actual = Backend.sqrt(x = input, e = 1e-5f)

        assertContentEquals(
            expected = DataBuffer.create(
                0.0031f,
                1.0000f,
                1.4142f,
                1.7320f,
                2.0000f,
                2.2360f,
                2.4494f,
                2.6457f,
                2.8284f,
                3.0000f,
            ),
            actual = actual,
        )
    }
}
