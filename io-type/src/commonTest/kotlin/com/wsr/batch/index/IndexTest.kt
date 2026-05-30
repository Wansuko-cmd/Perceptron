@file:Suppress("NonAsciiCharacters")

package com.wsr.batch.index

import com.wsr.assertContentEquals
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class IndexTest {
    @Test
    fun `D0gather_axis0=D0でD2バッチの行を収集`() = ioTypeTestRule {
        val d0 = IOType.d0(1f)
        val batch = batchOf(
            IOType.d2(2, 3) { i, j -> i * 3f + j },
            IOType.d2(2, 3) { i, j -> i * 3f + j + 6f },
        )
        val result = d0.gather(batch, axis = 0)
        assertContentEquals(IOType.d1(listOf(3f, 4f, 5f)), result[0])
        assertContentEquals(IOType.d1(listOf(9f, 10f, 11f)), result[1])
    }

    @Test
    fun `D0gather_axis1=D0でD2バッチの列を収集`() = ioTypeTestRule {
        val d0 = IOType.d0(1f)
        val batch = batchOf(
            IOType.d2(2, 3) { i, j -> i * 3f + j },
            IOType.d2(2, 3) { i, j -> i * 3f + j + 6f },
        )
        val result = d0.gather(batch, axis = 1)
        assertContentEquals(IOType.d1(listOf(1f, 4f)), result[0])
        assertContentEquals(IOType.d1(listOf(7f, 10f)), result[1])
    }

    @Test
    fun `D1Batchgather=インデックスでD2の行を収集`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(0f, 2f)),
            IOType.d1(listOf(1f, 0f)),
        )
        val d2 = IOType.d2(3, 2) { i, j -> i * 2f + j }
        val result = batch.gather(d2)
        assertContentEquals(IOType.d2(2, 2) { i, j -> i * 4f + j }, result[0])
        assertContentEquals(
            IOType.d2(listOf(2, 2), listOf(2f, 3f, 0f, 1f)),
            result[1],
        )
    }

    @Test
    fun `D2BatchscatterAdd=インデックスにバッチ値を散布加算`() = ioTypeTestRule {
        val batchD2 = batchOf(
            IOType.d2(2, 2) { i, j -> (i + 1) * 10f + (j + 1) },
            IOType.d2(2, 2) { i, j -> (i + 1) * 10f + (j + 1) + 100f },
        )
        val batchD1 = batchOf(
            IOType.d1(listOf(0f, 1f)),
            IOType.d1(listOf(0f, 1f)),
        )
        val result = batchD2.scatterAdd(batchD1, n = 2)
        assertContentEquals(
            IOType.d2(listOf(2, 2), listOf(122f, 124f, 142f, 144f)),
            result,
        )
    }
}
