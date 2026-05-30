@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.buffer.index

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.buffer.assertContentEquals
import com.wsr.knist.buffer.bufferTestRule
import kotlin.test.Test

class GatherTest {
    @Test
    fun `gather=Xの値をYの配列に置き換える`() = bufferTestRule {
        val x = DataBuffer.create(FloatArray(6) { it.toFloat() % 3 })
        val y = DataBuffer.create(FloatArray(24) { it.toFloat() })

        val actual = Backend.gather(
            x = x,
            y = y,
            i = 2,
            j = 3,
            k = 4,
        )

        assertContentEquals(
            expected = DataBuffer.create(
                floatArrayOf(
                    0f, 1f, 2f, 3f,
                    4f, 5f, 6f, 7f,

                    8f, 9f, 10f, 11f,
                    0f, 1f, 2f, 3f,

                    4f, 5f, 6f, 7f,
                    8f, 9f, 10f, 11f,

                    12f, 13f, 14f, 15f,
                    16f, 17f, 18f, 19f,

                    20f, 21f, 22f, 23f,
                    12f, 13f, 14f, 15f,

                    16f, 17f, 18f, 19f,
                    20f, 21f, 22f, 23f,
                ),
            ),
            actual = actual,
        )
    }
}
