@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.buffer.shape.fold

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.buffer.assertContentEquals
import com.wsr.knist.buffer.bufferTestRule
import kotlin.test.Test

class UnfoldTest {

    @Test
    fun `unfold_padding=0_stride=1=基本のスライド`() = bufferTestRule {
        val x = DataBuffer.create(1f, 2f, 3f, 4f, 5f)

        val actual = Backend.unfold(x = x, xi = 1, xj = 5, b = 1, window = 3, stride = 1, dilation = 1, padding = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                1f, 2f, 3f,
                2f, 3f, 4f,
                3f, 4f, 5f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `unfold_padding=1=両端が0埋めされる`() = bufferTestRule {
        val x = DataBuffer.create(1f, 2f, 3f, 4f)

        val actual = Backend.unfold(x = x, xi = 1, xj = 4, b = 1, window = 3, stride = 1, dilation = 1, padding = 1)

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f,
                1f, 2f, 3f,
                2f, 3f, 4f,
                3f, 4f, 0f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `unfold_stride=2=ウィンドウが飛び幅分ずれる`() = bufferTestRule {
        val x = DataBuffer.create(1f, 2f, 3f, 4f, 5f)

        val actual = Backend.unfold(x = x, xi = 1, xj = 5, b = 1, window = 3, stride = 2, dilation = 1, padding = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                2f,
                3f,
                3f,
                4f,
                5f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `unfold_dilation=2=タップの間隔が空く`() = bufferTestRule {
        val x = DataBuffer.create(1f, 2f, 3f, 4f, 5f)

        val actual = Backend.unfold(x = x, xi = 1, xj = 5, b = 1, window = 2, stride = 1, dilation = 2, padding = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                3f,
                2f,
                4f,
                3f,
                5f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `unfold_dilation=2_padding=1=dilationとpaddingを併用`() = bufferTestRule {
        val x = DataBuffer.create(1f, 2f, 3f, 4f, 5f)

        val actual = Backend.unfold(x = x, xi = 1, xj = 5, b = 1, window = 2, stride = 1, dilation = 2, padding = 1)

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 2f,
                1f, 3f,
                2f, 4f,
                3f, 5f,
                4f, 0f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `unfold_channel=複数=チャンネルごとに独立してスライド`() = bufferTestRule {
        val x = DataBuffer.create(1f, 2f, 3f, 4f, 5f, 6f)

        val actual = Backend.unfold(x = x, xi = 2, xj = 3, b = 1, window = 2, stride = 1, dilation = 1, padding = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                2f,
                2f,
                3f,
                4f,
                5f,
                5f,
                6f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `unfold_batch=複数=バッチごとに独立してスライド`() = bufferTestRule {
        val x = DataBuffer.create(1f, 2f, 3f, 4f, 5f, 6f)

        val actual = Backend.unfold(x = x, xi = 1, xj = 3, b = 2, window = 2, stride = 1, dilation = 1, padding = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                2f,
                2f,
                3f,
                4f,
                5f,
                5f,
                6f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `unfold_batch=複数_channel=複数_padding=1=複合ケース`() = bufferTestRule {
        val x = DataBuffer.create(
            1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f,
            9f, 10f, 11f, 12f,
            13f, 14f, 15f, 16f,
            17f, 18f, 19f, 20f,
            21f, 22f, 23f, 24f,
        )

        val actual = Backend.unfold(x = x, xi = 3, xj = 4, b = 2, window = 3, stride = 1, dilation = 1, padding = 1)

        assertContentEquals(
            expected = DataBuffer.create(
                0f, 1f, 2f,
                1f, 2f, 3f,
                2f, 3f, 4f,
                3f, 4f, 0f,

                0f, 5f, 6f,
                5f, 6f, 7f,
                6f, 7f, 8f,
                7f, 8f, 0f,

                0f, 9f, 10f,
                9f, 10f, 11f,
                10f, 11f, 12f,
                11f, 12f, 0f,

                0f, 13f, 14f,
                13f, 14f, 15f,
                14f, 15f, 16f,
                15f, 16f, 0f,

                0f, 17f, 18f,
                17f, 18f, 19f,
                18f, 19f, 20f,
                19f, 20f, 0f,

                0f, 21f, 22f,
                21f, 22f, 23f,
                22f, 23f, 24f,
                23f, 24f, 0f,
            ),
            actual = actual,
        )
    }
}
