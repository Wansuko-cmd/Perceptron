@file:Suppress("NonAsciiCharacters")

package com.wsr.core.shape

import com.wsr.assertContentEquals
import com.wsr.core.IOType
import com.wsr.core.d2
import com.wsr.core.d3
import com.wsr.core.shape.broadcastToD3
import com.wsr.core.shape.interleave
import com.wsr.core.shape.reshapeToD3
import com.wsr.core.shape.slice
import com.wsr.core.shape.transpose
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class ShapeD2Test {
    @Test
    fun `broadcastToD3_axis0=2次元を3次元にブロードキャスト`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val actual = d2.broadcastToD3(axis = 0, size = 2)
        assertContentEquals(
            expected = IOType.d3(2, 2, 2) { _, j, k -> j * 2f + k },
            actual = actual,
        )
    }

    @Test
    fun `broadcastToD3_axis1=2次元を3次元にブロードキャスト`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val actual = d2.broadcastToD3(axis = 1, size = 2)
        assertContentEquals(
            expected = IOType.d3(2, 2, 2) { i, _, k -> i * 2f + k },
            actual = actual,
        )
    }

    @Test
    fun `broadcastToD3_axis2=2次元を3次元にブロードキャスト`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val actual = d2.broadcastToD3(axis = 2, size = 2)
        assertContentEquals(
            expected = IOType.d3(2, 2, 2) { i, j, _ -> i * 2f + j },
            actual = actual,
        )
    }

    @Test
    fun `transpose=2次元転置`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.transpose()
        assertContentEquals(
            expected = IOType.d2(3, 2) { i, j -> j * 3f + i },
            actual = actual,
        )
    }

    @Test
    fun `reshapeToD3=2次元を3次元に変換`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 4) { i, j -> i * 4f + j }
        val actual = d2.reshapeToD3(2, 2, 2)
        assertContentEquals(
            expected = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            actual = actual,
        )
    }

    @Test
    fun `slice_axis0=指定範囲を2次元スライス`() = ioTypeTestRule {
        val d2 = IOType.d2(3, 3) { i, j -> i * 3f + j }
        val actual = d2.slice(0..1, axis = 0)
        assertContentEquals(
            expected = IOType.d2(2, 3) { i, j -> i * 3f + j },
            actual = actual,
        )
    }

    @Test
    fun `slice_axis1=指定範囲を2次元スライス`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.slice(0..1, axis = 1)
        assertContentEquals(
            expected = IOType.d2(2, 2) { i, j -> i * 3f + j },
            actual = actual,
        )
    }

    @Test
    fun `interleave_axis0=2つの2次元をインターリーブ`() = ioTypeTestRule {
        val d2a = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val d2b = IOType.d2(2, 2) { i, j -> 4f + i * 2f + j }
        val actual = d2a.interleave(d2b, axis = 0)
        assertContentEquals(
            expected = IOType.d2(listOf(4, 2), listOf(0f, 1f, 4f, 5f, 2f, 3f, 6f, 7f)),
            actual = actual,
        )
    }

    @Test
    fun `interleave_axis1=2つの2次元をインターリーブ`() = ioTypeTestRule {
        val d2a = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val d2b = IOType.d2(2, 2) { i, j -> 4f + i * 2f + j }
        val actual = d2a.interleave(d2b, axis = 1)
        assertContentEquals(
            expected = IOType.d2(listOf(2, 4), listOf(0f, 4f, 1f, 5f, 2f, 6f, 3f, 7f)),
            actual = actual,
        )
    }
}
