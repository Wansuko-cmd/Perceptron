@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.reduction

import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.batchOf
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class D2ExtTest {
    @Test
    fun `sum=D2バッチの各要素の合計`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.sum()
        assertContentEquals(IOType.d0(6f), result[0])
        assertContentEquals(IOType.d0(22f), result[1])
    }

    @Test
    fun `sum_axis0=D2バッチのaxis0合計`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.sum(axis = 0)
        assertContentEquals(IOType.d1(listOf(2f, 4f)), result[0])
        assertContentEquals(IOType.d1(listOf(10f, 12f)), result[1])
    }

    @Test
    fun `sum_axis1=D2バッチのaxis1合計`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.sum(axis = 1)
        assertContentEquals(IOType.d1(listOf(1f, 5f)), result[0])
        assertContentEquals(IOType.d1(listOf(9f, 13f)), result[1])
    }

    @Test
    fun `max=D2バッチの各要素の最大値`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.max()
        assertContentEquals(IOType.d0(3f), result[0])
        assertContentEquals(IOType.d0(7f), result[1])
    }

    @Test
    fun `min=D2バッチの各要素の最小値`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.min()
        assertContentEquals(IOType.d0(0f), result[0])
        assertContentEquals(IOType.d0(4f), result[1])
    }
}
