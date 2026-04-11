@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.transpose

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.create
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import kotlin.test.Test

class TransposeD2Test {
    val input = DataBuffer.create(FloatArray(20) { it.toFloat() })

    @Test
    fun `transpose=2次元転置`() = bufferTestRule {
        val result = Backend.transpose(x = input, xi = 4, xj = 5)

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 5f, 10f, 15f,
                1f, 6f, 11f, 16f,
                2f, 7f, 12f, 17f,
                3f, 8f, 13f, 18f,
                4f, 9f, 14f, 19f,
            ),
            actual = result,
        )
    }
}
