@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.buffer.reduction

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.buffer.assertContentEquals
import com.wsr.knist.buffer.bufferTestRule
import kotlin.test.Test

class TopKTest {
    @Test
    fun `2次元topK_k=1_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.topK(x = x, xi = 4, xj = 6, k = 1, axis = 0, random = kotlin.random.Random)

        assertContentEquals(expected = DataBuffer.create(3f, 3f, 3f, 3f, 3f, 3f), actual = actual)
    }

    @Test
    fun `2次元topK_k=1_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.topK(x = x, xi = 4, xj = 6, k = 1, axis = 1, random = kotlin.random.Random)

        assertContentEquals(expected = DataBuffer.create(5f, 5f, 5f, 5f), actual = actual)
    }

    @Test
    fun `3次元topK_k=1_axis=0`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.topK(x = x, xi = 4, xj = 3, xk = 2, k = 1, axis = 0, random = kotlin.random.Random)

        assertContentEquals(expected = DataBuffer.create(3f, 3f, 3f, 3f, 3f, 3f), actual = actual)
    }

    @Test
    fun `3次元topK_k=1_axis=1`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.topK(x = x, xi = 4, xj = 3, xk = 2, k = 1, axis = 1, random = kotlin.random.Random)

        assertContentEquals(expected = DataBuffer.create(2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f), actual = actual)
    }

    @Test
    fun `3次元topK_k=1_axis=2`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.topK(x = x, xi = 4, xj = 3, xk = 2, k = 1, axis = 2, random = kotlin.random.Random)

        assertContentEquals(
            expected = DataBuffer.create(FloatArray(12) { 1f }),
            actual = actual,
        )
    }
}
