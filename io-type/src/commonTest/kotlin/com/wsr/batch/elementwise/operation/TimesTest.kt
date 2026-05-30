@file:Suppress("NonAsciiCharacters")

package com.wsr.batch.elementwise.operation

import com.wsr.assertContentEquals
import com.wsr.batch.batchOf
import com.wsr.batch.elementwise.operation.times.times
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.core.d1
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class TimesTest {
    @Test
    fun `D0バッチ×スカラー`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(2f), IOType.d0(3f), IOType.d0(4f))
        val result = batch * 10f
        assertContentEquals(IOType.d0(20f), result[0])
        assertContentEquals(IOType.d0(30f), result[1])
        assertContentEquals(IOType.d0(40f), result[2])
    }

    @Test
    fun `スカラー×D0バッチ`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(2f), IOType.d0(3f), IOType.d0(4f))
        val result = 10f * batch
        assertContentEquals(IOType.d0(20f), result[0])
        assertContentEquals(IOType.d0(30f), result[1])
        assertContentEquals(IOType.d0(40f), result[2])
    }

    @Test
    fun `D0バッチ×D0バッチ`() = ioTypeTestRule {
        val batch1 = batchOf(IOType.d0(2f), IOType.d0(3f))
        val batch2 = batchOf(IOType.d0(4f), IOType.d0(5f))
        val result = batch1 * batch2
        assertContentEquals(IOType.d0(8f), result[0])
        assertContentEquals(IOType.d0(15f), result[1])
    }

    @Test
    fun `D1バッチ×スカラー`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = batch * 2f
        assertContentEquals(IOType.d1(listOf(2f, 4f, 6f)), result[0])
        assertContentEquals(IOType.d1(listOf(8f, 10f, 12f)), result[1])
    }

    @Test
    fun `D1バッチ×D1バッチ`() = ioTypeTestRule {
        val batch1 = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val batch2 = batchOf(
            IOType.d1(listOf(2f, 2f, 2f)),
            IOType.d1(listOf(3f, 3f, 3f)),
        )
        val result = batch1 * batch2
        assertContentEquals(IOType.d1(listOf(2f, 4f, 6f)), result[0])
        assertContentEquals(IOType.d1(listOf(12f, 15f, 18f)), result[1])
    }

    @Test
    fun `D1×D1バッチ=ブロードキャスト`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(2f, 3f, 4f))
        val batch = batchOf(
            IOType.d1(listOf(1f, 1f, 1f)),
            IOType.d1(listOf(2f, 2f, 2f)),
        )
        val result = d1 * batch
        assertContentEquals(IOType.d1(listOf(2f, 3f, 4f)), result[0])
        assertContentEquals(IOType.d1(listOf(4f, 6f, 8f)), result[1])
    }
}
