@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.reduction

import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.batchOf
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class D3ExtTest {
    @Test
    fun `sum=D3バッチの全合計`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 8f },
        )
        val result = batch.sum()
        assertContentEquals(IOType.d0(28f), result[0])
        assertContentEquals(IOType.d0(92f), result[1])
    }

    @Test
    fun `sum_axis0=D3バッチのaxis0合計`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 8f },
        )
        val result = batch.sum(axis = 0)
        assertContentEquals(IOType.d2(2, 2) { i, j -> i * 4f + j * 2f + 4f }, result[0])
        assertContentEquals(IOType.d2(2, 2) { i, j -> i * 4f + j * 2f + 20f }, result[1])
    }

    @Test
    fun `max=D3バッチの最大値`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 8f },
        )
        val result = batch.max()
        assertContentEquals(IOType.d0(7f), result[0])
        assertContentEquals(IOType.d0(15f), result[1])
    }

    @Test
    fun `min=D3バッチの最小値`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 8f },
        )
        val result = batch.min()
        assertContentEquals(IOType.d0(0f), result[0])
        assertContentEquals(IOType.d0(8f), result[1])
    }

    private val d3v0 = IOType.d3(shape = listOf(2, 3, 2), value = listOf(
        1f, 2f,
        5f, 1f,
        0f, 3f,

        4f, 0f,
        2f, 4f,
        3f, 1f,
    ))

    private val d3v1 = IOType.d3(shape = listOf(2, 3, 2), value = listOf(
        0f, 4f,
        3f, 2f,
        6f, 1f,

        5f, 0f,
        1f, 3f,
        2f, 4f,
    ))

    @Test
    fun `maxIndex_axis0=D3バッチのaxis0最大値インデックス`() = ioTypeTestRule {
        val batch = batchOf(d3v0, d3v1)
        val result = batch.maxIndex(axis = 0)
        assertContentEquals(IOType.d2(shape = listOf(3, 2), value = listOf(1f, 0f, 0f, 1f, 1f, 0f)), result[0])
        assertContentEquals(IOType.d2(shape = listOf(3, 2), value = listOf(1f, 0f, 0f, 1f, 0f, 1f)), result[1])
    }

    @Test
    fun `maxIndex_axis1=D3バッチのaxis1最大値インデックス`() = ioTypeTestRule {
        val batch = batchOf(d3v0, d3v1)
        val result = batch.maxIndex(axis = 1)
        assertContentEquals(IOType.d2(shape = listOf(2, 2), value = listOf(1f, 2f, 0f, 1f)), result[0])
        assertContentEquals(IOType.d2(shape = listOf(2, 2), value = listOf(2f, 0f, 0f, 2f)), result[1])
    }

    @Test
    fun `maxIndex_axis2=D3バッチのaxis2最大値インデックス`() = ioTypeTestRule {
        val batch = batchOf(d3v0, d3v1)
        val result = batch.maxIndex(axis = 2)
        assertContentEquals(IOType.d2(shape = listOf(2, 3), value = listOf(1f, 0f, 1f, 0f, 1f, 0f)), result[0])
        assertContentEquals(IOType.d2(shape = listOf(2, 3), value = listOf(1f, 0f, 0f, 0f, 1f, 1f)), result[1])
    }
}
