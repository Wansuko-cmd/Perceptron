@file:Suppress("NonAsciiCharacters")

package com.wsr.buffer.index

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.buffer.assertContentEquals
import com.wsr.buffer.bufferTestRule
import com.wsr.create
import kotlin.test.Test

class ScatterAddTest {
    @Test
    fun `scatterAdd=Yの値を元にXを圧縮する`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(48) { it.toFloat() })
        val y = DataBuffer.create(FloatArray(6) { it.toFloat() % 3 })

        val actual = Backend.scatterAdd(
            x = x,
            y = y,
            i = 2,
            j = 3,
            k = 4,
            b = 1,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                floatArrayOf(
                    12f, 14f, 16f, 18f,
                    20f, 22f, 24f, 26f,
                    28f, 30f, 32f, 34f,

                    60f, 62f, 64f, 66f,
                    68f, 70f, 72f, 74f,
                    76f, 78f, 80f, 82f,
                ),
            ),
            actual = actual,
        )
    }
}
