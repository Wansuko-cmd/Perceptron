@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.compare

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import kotlin.test.Test

class EqualsTest {
    @Test
    fun `equals=入力のFloat値と等価かどうか`() = bufferTestRule {
        val input = DataBuffer.create(1f, 0f, 0.95f, 0.8f, 1.05f, 1.2f)

        val actual = Backend.equals(
            x = input,
            y = 1f,
            absoluteTolerance = 0f,
            relativeTolerance = 0f,
        )

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(1f, 0f, 0f, 0f, 0f, 0f)),
            actual = actual,
        )
    }

    @Test
    fun `equals=入力のFloat値と絶対誤差内かどうか`() = bufferTestRule {
        val input = DataBuffer.create(1f, 0f, 0.95f, 0.8f, 1.05f, 1.2f)

        val actual = Backend.equals(
            x = input,
            y = 1f,
            absoluteTolerance = 0.1f,
            relativeTolerance = 0f,
        )

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(1f, 0f, 1f, 0f, 1f, 0f)),
            actual = actual,
        )
    }

    @Test
    fun `equals=入力のFloat値と相対誤差内かどうか`() = bufferTestRule {
        val input = DataBuffer.create(1f, 0f, 0.95f, 0.8f, 1.05f, 1.2f)

        val actual = Backend.equals(
            x = input,
            y = 1f,
            absoluteTolerance = 0f,
            relativeTolerance = 0.1f,
        )

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(1f, 0f, 1f, 0f, 1f, 0f)),
            actual = actual,
        )
    }

    @Test
    fun `equals=入力のDataBufferと等価かどうか`() = bufferTestRule {
        val input = DataBuffer.create(1f, 0f, 0.95f, 0.8f, 1.05f, 1.2f)
        val other = DataBuffer.create(1f, 1f, 1f, 1f, 1f, 1f)

        val actual = Backend.equals(
            x = input,
            y = other,
            absoluteTolerance = 0f,
            relativeTolerance = 0f,
        )

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(1f, 0f, 0f, 0f, 0f, 0f)),
            actual = actual,
        )
    }

    @Test
    fun `equals=入力のDataBufferと絶対誤差内かどうか`() = bufferTestRule {
        val input = DataBuffer.create(1f, 0f, 0.95f, 0.8f, 1.05f, 1.2f)
        val other = DataBuffer.create(1f, 1f, 1f, 1f, 1f, 1f)

        val actual = Backend.equals(
            x = input,
            y = other,
            absoluteTolerance = 0.1f,
            relativeTolerance = 0f,
        )

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(1f, 0f, 1f, 0f, 1f, 0f)),
            actual = actual,
        )
    }

    @Test
    fun `equals=入力のDataBufferと相対誤差内かどうか`() = bufferTestRule {
        val input = DataBuffer.create(1f, 0f, 0.95f, 0.8f, 1.05f, 1.2f)
        val other = DataBuffer.create(1f, 1f, 1f, 1f, 1f, 1f)

        val actual = Backend.equals(
            x = input,
            y = other,
            absoluteTolerance = 0f,
            relativeTolerance = 0.1f,
        )

        assertContentEquals(
            expected = DataBuffer.create(floatArrayOf(1f, 0f, 1f, 0f, 1f, 0f)),
            actual = actual,
        )
    }
}
