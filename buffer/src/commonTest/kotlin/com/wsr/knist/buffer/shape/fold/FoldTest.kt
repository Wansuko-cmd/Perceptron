@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.buffer.shape.fold

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.buffer.assertContentEquals
import com.wsr.knist.buffer.bufferTestRule
import kotlin.test.Test

class FoldTest {

    @Test
    fun `fold_padding=0_stride=1=基本の畳み込み`() = bufferTestRule {
        val x = DataBuffer.create(
            1f, 2f, 3f,
            2f, 3f, 4f,
            3f, 4f, 5f,
        )

        val actual = Backend.fold(x = x, xi = 1, xj = 3, xk = 3, b = 1, stride = 1, dilation = 1, padding = 0)

        assertContentEquals(
            expected = DataBuffer.create(1f, 4f, 9f, 8f, 5f),
            actual = actual,
        )
    }

    @Test
    fun `fold_padding=1=両端のパディング分は無視される`() = bufferTestRule {
        val x = DataBuffer.create(
            0f, 1f, 2f,
            1f, 2f, 3f,
            2f, 3f, 4f,
            3f, 4f, 0f,
        )

        val actual = Backend.fold(x = x, xi = 1, xj = 4, xk = 3, b = 1, stride = 1, dilation = 1, padding = 1)

        assertContentEquals(
            expected = DataBuffer.create(2f, 6f, 9f, 8f),
            actual = actual,
        )
    }

    @Test
    fun `fold_stride=2=ウィンドウが飛び幅分ずれる`() = bufferTestRule {
        val x = DataBuffer.create(
            1f,
            2f,
            3f,
            3f,
            4f,
            5f,
        )

        val actual = Backend.fold(x = x, xi = 1, xj = 2, xk = 3, b = 1, stride = 2, dilation = 1, padding = 0)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 6f, 4f, 5f),
            actual = actual,
        )
    }

    @Test
    fun `fold_dilation=2=タップの間隔を空けて復元`() = bufferTestRule {
        val x = DataBuffer.create(
            1f,
            3f,
            2f,
            4f,
            3f,
            5f,
        )

        val actual = Backend.fold(x = x, xi = 1, xj = 3, xk = 2, b = 1, stride = 1, dilation = 2, padding = 0)

        assertContentEquals(
            expected = DataBuffer.create(1f, 2f, 6f, 4f, 5f),
            actual = actual,
        )
    }

    @Test
    fun `fold_dilation=2_padding=1=dilationとpaddingを併用`() = bufferTestRule {
        val x = DataBuffer.create(
            1f,
            2f,
            3f,
            4f,
            5f,
            6f,
        )

        val actual = Backend.fold(x = x, xi = 1, xj = 3, xk = 2, b = 1, stride = 1, dilation = 2, padding = 1)

        assertContentEquals(
            expected = DataBuffer.create(3f, 7f, 4f),
            actual = actual,
        )
    }

    @Test
    fun `fold_channel=複数=チャンネルごとに独立して畳み込む`() = bufferTestRule {
        val x = DataBuffer.create(
            1f,
            2f,
            2f,
            3f,
            4f,
            5f,
            5f,
            6f,
        )

        val actual = Backend.fold(x = x, xi = 2, xj = 2, xk = 2, b = 1, stride = 1, dilation = 1, padding = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                4f,
                3f,
                4f,
                10f,
                6f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `fold_batch=複数=バッチごとに独立して畳み込む`() = bufferTestRule {
        val x = DataBuffer.create(
            1f,
            2f,
            2f,
            3f,
            4f,
            5f,
            5f,
            6f,
        )

        val actual = Backend.fold(x = x, xi = 1, xj = 2, xk = 2, b = 2, stride = 1, dilation = 1, padding = 0)

        assertContentEquals(
            expected = DataBuffer.create(
                1f,
                4f,
                3f,
                4f,
                10f,
                6f,
            ),
            actual = actual,
        )
    }

    @Test
    fun `fold_batch=複数_channel=複数_padding=1=複合ケース`() = bufferTestRule {
        val x = DataBuffer.create(
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
        )

        val actual = Backend.fold(x = x, xi = 3, xj = 4, xk = 3, b = 2, stride = 1, dilation = 1, padding = 1)

        assertContentEquals(
            expected = DataBuffer.create(
                2f, 6f, 9f, 8f,
                10f, 18f, 21f, 16f,
                18f, 30f, 33f, 24f,
                26f, 42f, 45f, 32f,
                34f, 54f, 57f, 40f,
                42f, 66f, 69f, 48f,
            ),
            actual = actual,
        )
    }
}
