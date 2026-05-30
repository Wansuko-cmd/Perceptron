@file:Suppress("NonAsciiCharacters")

package com.wsr.batch.shape

import com.wsr.assertContentEquals
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.batch.i
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.ioTypeTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class D1ExtTest {
    @Test
    fun `broadcastToD2_axis0=D1バッチを2次元にブロードキャスト`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = batch.broadcastToD2(axis = 0, size = 2)
        assertContentEquals(IOType.d2(2, 3) { _, j -> j + 1f }, result[0])
        assertContentEquals(IOType.d2(2, 3) { _, j -> j + 4f }, result[1])
    }

    @Test
    fun `toD2=D1バッチを2次元に変換`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = batch.toD2()
        assertContentEquals(
            IOType.d2(2, 3) { i, j -> i * 3f + j + 1f },
            result,
        )
    }

    @Test
    fun `D2toBatch=2次元をD1バッチに変換`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j + 1f }
        val batch = d2.toBatch()
        assertEquals(2, batch.size)
        assertEquals(3, batch.i)
        assertContentEquals(IOType.d1(listOf(1f, 2f, 3f)), batch[0])
        assertContentEquals(IOType.d1(listOf(4f, 5f, 6f)), batch[1])
    }

    @Test
    fun `reshapeToD2=D1バッチをD2バッチに変形`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(0f, 1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f, 7f)),
        )
        val result = batch.reshapeToD2(2, 2)
        assertContentEquals(IOType.d2(2, 2) { i, j -> i * 2f + j }, result[0])
        assertContentEquals(IOType.d2(2, 2) { i, j -> i * 2f + j + 4f }, result[1])
    }

    @Test
    fun `slice=D1バッチの部分抽出`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(0f, 1f, 2f, 3f, 4f)),
            IOType.d1(listOf(5f, 6f, 7f, 8f, 9f)),
        )
        val result = batch.slice(1..3)
        assertContentEquals(IOType.d1(listOf(1f, 2f, 3f)), result[0])
        assertContentEquals(IOType.d1(listOf(6f, 7f, 8f)), result[1])
    }

    @Test
    fun `interleave=2つのD1バッチをインターリーブ`() = ioTypeTestRule {
        val batch1 = batchOf(
            IOType.d1(listOf(0f, 2f)),
            IOType.d1(listOf(4f, 6f)),
        )
        val batch2 = batchOf(
            IOType.d1(listOf(1f, 3f)),
            IOType.d1(listOf(5f, 7f)),
        )
        val result = batch1.interleave(batch2)
        assertContentEquals(IOType.d1(listOf(0f, 1f, 2f, 3f)), result[0])
        assertContentEquals(IOType.d1(listOf(4f, 5f, 6f, 7f)), result[1])
    }
}
