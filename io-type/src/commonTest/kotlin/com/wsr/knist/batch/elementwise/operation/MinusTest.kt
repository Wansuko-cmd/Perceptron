@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.elementwise.operation

import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.batchOf
import com.wsr.knist.batch.elementwise.operation.minus.minus
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class MinusTest {
    @Test
    fun `D0バッチ-スカラー`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(11f), IOType.d0(12f), IOType.d0(13f))
        val result = batch - 10f
        assertContentEquals(IOType.d0(1f), result[0])
        assertContentEquals(IOType.d0(2f), result[1])
        assertContentEquals(IOType.d0(3f), result[2])
    }

    @Test
    fun `スカラー-D0バッチ`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val result = 10f - batch
        assertContentEquals(IOType.d0(9f), result[0])
        assertContentEquals(IOType.d0(8f), result[1])
        assertContentEquals(IOType.d0(7f), result[2])
    }

    @Test
    fun `D0バッチ-D0バッチ`() = ioTypeTestRule {
        val batch1 = batchOf(IOType.d0(10f), IOType.d0(20f))
        val batch2 = batchOf(IOType.d0(1f), IOType.d0(2f))
        val result = batch1 - batch2
        assertContentEquals(IOType.d0(9f), result[0])
        assertContentEquals(IOType.d0(18f), result[1])
    }

    @Test
    fun `D1バッチ-スカラー`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(11f, 12f, 13f)),
            IOType.d1(listOf(14f, 15f, 16f)),
        )
        val result = batch - 10f
        assertContentEquals(IOType.d1(listOf(1f, 2f, 3f)), result[0])
        assertContentEquals(IOType.d1(listOf(4f, 5f, 6f)), result[1])
    }

    @Test
    fun `D1バッチ-D1バッチ`() = ioTypeTestRule {
        val batch1 = batchOf(
            IOType.d1(listOf(10f, 20f, 30f)),
            IOType.d1(listOf(40f, 50f, 60f)),
        )
        val batch2 = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = batch1 - batch2
        assertContentEquals(IOType.d1(listOf(9f, 18f, 27f)), result[0])
        assertContentEquals(IOType.d1(listOf(36f, 45f, 54f)), result[1])
    }

    @Test
    fun `D1バッチ-D1=ブロードキャスト`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(11f, 12f, 13f)),
            IOType.d1(listOf(14f, 15f, 16f)),
        )
        val d1 = IOType.d1(listOf(1f, 2f, 3f))
        val result = batch - d1
        assertContentEquals(IOType.d1(listOf(10f, 10f, 10f)), result[0])
        assertContentEquals(IOType.d1(listOf(13f, 13f, 13f)), result[1])
    }
}
