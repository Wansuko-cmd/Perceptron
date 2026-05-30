@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.reduction.average

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
    fun `average=D2バッチの各要素の平均`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.average()
        assertContentEquals(IOType.d0(1.5f), result[0])
        assertContentEquals(IOType.d0(5.5f), result[1])
    }

    @Test
    fun `average_axis0=D2バッチのaxis0平均`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.average(axis = 0)
        assertContentEquals(IOType.d1(listOf(1f, 2f)), result[0])
        assertContentEquals(IOType.d1(listOf(5f, 6f)), result[1])
    }

    @Test
    fun `average_axis1=D2バッチのaxis1平均`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.average(axis = 1)
        assertContentEquals(IOType.d1(listOf(0.5f, 2.5f)), result[0])
        assertContentEquals(IOType.d1(listOf(4.5f, 6.5f)), result[1])
    }

    @Test
    fun `batchAverage=D2バッチのバッチ平均`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.batchAverage()
        assertContentEquals(IOType.d2(2, 2) { i, j -> i * 2f + j + 2f }, result)
    }
}
