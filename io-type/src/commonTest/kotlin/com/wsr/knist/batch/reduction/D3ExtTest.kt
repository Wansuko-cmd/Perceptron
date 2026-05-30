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
}
