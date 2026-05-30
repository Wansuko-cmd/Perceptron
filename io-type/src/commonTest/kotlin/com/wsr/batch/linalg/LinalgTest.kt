@file:Suppress("NonAsciiCharacters")

package com.wsr.batch.linalg

import com.wsr.assertContentEquals
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class LinalgTest {
    @Test
    fun `D1バッチ内積=バッチごとに内積を計算`() = ioTypeTestRule {
        val batch1 = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val batch2 = batchOf(
            IOType.d1(listOf(1f, 0f, 0f)),
            IOType.d1(listOf(0f, 1f, 0f)),
        )
        val result = batch1 inner batch2
        assertContentEquals(IOType.d0(1f), result[0])
        assertContentEquals(IOType.d0(5f), result[1])
    }

    @Test
    fun `D2 matMul D1バッチ=行列とバッチベクトルの積`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val batch = batchOf(
            IOType.d1(listOf(1f, 0f, 0f)),
            IOType.d1(listOf(0f, 1f, 0f)),
        )
        val result = d2.matMul(batch)
        assertContentEquals(IOType.d1(listOf(0f, 3f)), result[0])
        assertContentEquals(IOType.d1(listOf(1f, 4f)), result[1])
    }

    @Test
    fun `D2バッチ matMul D2=バッチ行列積`() = ioTypeTestRule {
        val batchD2 = batchOf(
            IOType.d2(2, 3) { i, j -> i * 3f + j },
            IOType.d2(2, 3) { i, j -> i * 3f + j + 6f },
        )
        val d2 = IOType.d2(3, 2) { i, j -> i * 2f + j }
        val result = batchD2.matMul(d2)
        assertContentEquals(
            IOType.d2(listOf(2, 2), listOf(10f, 13f, 28f, 40f)),
            result[0],
        )
        assertContentEquals(
            IOType.d2(listOf(2, 2), listOf(46f, 67f, 64f, 94f)),
            result[1],
        )
    }

    @Test
    fun `D2バッチ matMul D2バッチ=バッチごとの行列積`() = ioTypeTestRule {
        val batchA = batchOf(
            IOType.d2(2, 3) { i, j -> i * 3f + j },
            IOType.d2(2, 3) { i, j -> i * 3f + j + 6f },
        )
        val batchB = batchOf(
            IOType.d2(3, 2) { i, j -> i * 2f + j },
            IOType.d2(3, 2) { i, j -> i * 2f + j },
        )
        val result = batchA.matMul(batchB)
        assertContentEquals(
            IOType.d2(listOf(2, 2), listOf(10f, 13f, 28f, 40f)),
            result[0],
        )
        assertContentEquals(
            IOType.d2(listOf(2, 2), listOf(46f, 67f, 64f, 94f)),
            result[1],
        )
    }
}
