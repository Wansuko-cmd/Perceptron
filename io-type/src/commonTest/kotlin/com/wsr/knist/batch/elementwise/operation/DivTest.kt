@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.elementwise.operation

import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.batchOf
import com.wsr.knist.batch.elementwise.operation.div.div
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class DivTest {
    @Test
    fun `D0バッチ÷スカラー`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(10f), IOType.d0(20f), IOType.d0(30f))
        val result = batch / 10f
        assertContentEquals(IOType.d0(1f), result[0])
        assertContentEquals(IOType.d0(2f), result[1])
        assertContentEquals(IOType.d0(3f), result[2])
    }

    @Test
    fun `スカラー÷D0バッチ`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(2f), IOType.d0(4f), IOType.d0(5f))
        val result = 20f / batch
        assertContentEquals(IOType.d0(10f), result[0])
        assertContentEquals(IOType.d0(5f), result[1])
        assertContentEquals(IOType.d0(4f), result[2])
    }

    @Test
    fun `D0バッチ÷D0バッチ`() = ioTypeTestRule {
        val batch1 = batchOf(IOType.d0(10f), IOType.d0(20f))
        val batch2 = batchOf(IOType.d0(2f), IOType.d0(4f))
        val result = batch1 / batch2
        assertContentEquals(IOType.d0(5f), result[0])
        assertContentEquals(IOType.d0(5f), result[1])
    }

    @Test
    fun `D1バッチ÷スカラー`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(2f, 4f, 6f)),
            IOType.d1(listOf(8f, 10f, 12f)),
        )
        val result = batch / 2f
        assertContentEquals(IOType.d1(listOf(1f, 2f, 3f)), result[0])
        assertContentEquals(IOType.d1(listOf(4f, 5f, 6f)), result[1])
    }

    @Test
    fun `D1バッチ÷D1バッチ`() = ioTypeTestRule {
        val batch1 = batchOf(
            IOType.d1(listOf(2f, 4f, 6f)),
            IOType.d1(listOf(8f, 10f, 12f)),
        )
        val batch2 = batchOf(
            IOType.d1(listOf(2f, 2f, 2f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = batch1 / batch2
        assertContentEquals(IOType.d1(listOf(1f, 2f, 3f)), result[0])
        assertContentEquals(IOType.d1(listOf(2f, 2f, 2f)), result[1])
    }

    @Test
    fun `D1バッチ÷D1=ブロードキャスト`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(2f, 4f, 6f)),
            IOType.d1(listOf(8f, 10f, 12f)),
        )
        val d1 = IOType.d1(listOf(2f, 2f, 2f))
        val result = batch / d1
        assertContentEquals(IOType.d1(listOf(1f, 2f, 3f)), result[0])
        assertContentEquals(IOType.d1(listOf(4f, 5f, 6f)), result[1])
    }
}
