@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.reduction.average
import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class D3ExtTest {
    @Test
    fun `average=D3バッチの各要素の平均`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 8f },
        )
        val result = batch.average()
        assertContentEquals(IOType.d0(3.5f), result[0])
        assertContentEquals(IOType.d0(11.5f), result[1])
    }

    @Test
    fun `average_axis0=D3バッチのaxis0平均`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 8f },
        )
        val result = batch.average(axis = 0)
        assertContentEquals(IOType.d2(2, 2) { i, j -> i * 2f + j + 2f }, result[0])
        assertContentEquals(IOType.d2(2, 2) { i, j -> i * 2f + j + 10f }, result[1])
    }

    @Test
    fun `batchAverage=D3バッチのバッチ平均`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 8f },
        )
        val result = batch.batchAverage()
        assertContentEquals(
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 4f },
            result,
        )
    }
}
