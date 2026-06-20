@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.shape
import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class D4ExtTest {
    @Test
    fun `concat_axis0=D4バッチをaxis0で連結`() = ioTypeTestRule {
        val batch1 = Batch.of(
            IOType.d4(1, 2, 2, 2) { _, j, k, l -> j * 4f + k * 2f + l },
            IOType.d4(1, 2, 2, 2) { _, j, k, l -> j * 4f + k * 2f + l + 16f },
        )
        val batch2 = Batch.of(
            IOType.d4(1, 2, 2, 2) { _, j, k, l -> j * 4f + k * 2f + l + 8f },
            IOType.d4(1, 2, 2, 2) { _, j, k, l -> j * 4f + k * 2f + l + 24f },
        )
        val result = batch1.concat(batch2, axis = 0)
        assertContentEquals(IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }, result[0])
        assertContentEquals(IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l + 16f }, result[1])
    }

    @Test
    fun `concat_axis1=D4バッチをaxis1で連結`() = ioTypeTestRule {
        val batch1 = Batch.of(
            IOType.d4(2, 1, 2, 2) { i, _, k, l -> i * 8f + k * 2f + l },
            IOType.d4(2, 1, 2, 2) { i, _, k, l -> i * 8f + k * 2f + l + 16f },
        )
        val batch2 = Batch.of(
            IOType.d4(2, 1, 2, 2) { i, _, k, l -> i * 8f + k * 2f + l + 4f },
            IOType.d4(2, 1, 2, 2) { i, _, k, l -> i * 8f + k * 2f + l + 20f },
        )
        val result = batch1.concat(batch2, axis = 1)
        assertContentEquals(IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }, result[0])
        assertContentEquals(IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l + 16f }, result[1])
    }

    @Test
    fun `concat_axis2=D4バッチをaxis2で連結`() = ioTypeTestRule {
        val batch1 = Batch.of(
            IOType.d4(2, 2, 1, 2) { i, j, _, l -> i * 8f + j * 4f + l },
            IOType.d4(2, 2, 1, 2) { i, j, _, l -> i * 8f + j * 4f + l + 16f },
        )
        val batch2 = Batch.of(
            IOType.d4(2, 2, 1, 2) { i, j, _, l -> i * 8f + j * 4f + l + 2f },
            IOType.d4(2, 2, 1, 2) { i, j, _, l -> i * 8f + j * 4f + l + 18f },
        )
        val result = batch1.concat(batch2, axis = 2)
        assertContentEquals(IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }, result[0])
        assertContentEquals(IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l + 16f }, result[1])
    }

    @Test
    fun `concat_axis3=D4バッチをaxis3で連結`() = ioTypeTestRule {
        val batch1 = Batch.of(
            IOType.d4(2, 2, 2, 1) { i, j, k, _ -> i * 8f + j * 4f + k * 2f },
            IOType.d4(2, 2, 2, 1) { i, j, k, _ -> i * 8f + j * 4f + k * 2f + 16f },
        )
        val batch2 = Batch.of(
            IOType.d4(2, 2, 2, 1) { i, j, k, _ -> i * 8f + j * 4f + k * 2f + 1f },
            IOType.d4(2, 2, 2, 1) { i, j, k, _ -> i * 8f + j * 4f + k * 2f + 17f },
        )
        val result = batch1.concat(batch2, axis = 3)
        assertContentEquals(IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }, result[0])
        assertContentEquals(IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l + 16f }, result[1])
    }

    @Test
    fun `reshapeToD3=D4バッチをD3バッチに変形`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l },
            IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l + 16f },
        )
        val result = batch.reshapeToD3(4, 2, 2)
        assertEquals(2, result.size)
        assertEquals(listOf(4, 2, 2), result.shape)
        assertContentEquals(IOType.d3(4, 2, 2) { i, j, k -> i * 4f + j * 2f + k }, result[0])
        assertContentEquals(IOType.d3(4, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 16f }, result[1])
    }
}
