@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.elementwise.compare
import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class D0ExtTest {
    @Test
    fun `eq_Float=D0バッチとFloat値の等値比較`() = ioTypeTestRule {
        val batch = Batch.of(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val result = batch eq 2f
        assertContentEquals(IOType.d0(0f), result[0])
        assertContentEquals(IOType.d0(1f), result[1])
        assertContentEquals(IOType.d0(0f), result[2])
    }

    @Test
    fun `eq_D0s=D0バッチ同士の等値比較`() = ioTypeTestRule {
        val batch1 = Batch.of(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val batch2 = Batch.of(IOType.d0(1f), IOType.d0(0f), IOType.d0(3f))
        val result = batch1 eq batch2
        assertContentEquals(IOType.d0(1f), result[0])
        assertContentEquals(IOType.d0(0f), result[1])
        assertContentEquals(IOType.d0(1f), result[2])
    }

    @Test
    fun `gt_Float=D0バッチとFloat値の大比較`() = ioTypeTestRule {
        val batch = Batch.of(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val result = batch gt 2f
        assertContentEquals(IOType.d0(0f), result[0])
        assertContentEquals(IOType.d0(0f), result[1])
        assertContentEquals(IOType.d0(1f), result[2])
    }

    @Test
    fun `gt_D0s=D0バッチ同士の大比較`() = ioTypeTestRule {
        val batch1 = Batch.of(IOType.d0(1f), IOType.d0(3f), IOType.d0(2f))
        val batch2 = Batch.of(IOType.d0(2f), IOType.d0(2f), IOType.d0(2f))
        val result = batch1 gt batch2
        assertContentEquals(IOType.d0(0f), result[0])
        assertContentEquals(IOType.d0(1f), result[1])
        assertContentEquals(IOType.d0(0f), result[2])
    }

    @Test
    fun `lt_Float=D0バッチとFloat値の小比較`() = ioTypeTestRule {
        val batch = Batch.of(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val result = batch lt 2f
        assertContentEquals(IOType.d0(1f), result[0])
        assertContentEquals(IOType.d0(0f), result[1])
        assertContentEquals(IOType.d0(0f), result[2])
    }

    @Test
    fun `lt_D0s=D0バッチ同士の小比較`() = ioTypeTestRule {
        val batch1 = Batch.of(IOType.d0(1f), IOType.d0(3f), IOType.d0(2f))
        val batch2 = Batch.of(IOType.d0(2f), IOType.d0(2f), IOType.d0(2f))
        val result = batch1 lt batch2
        assertContentEquals(IOType.d0(1f), result[0])
        assertContentEquals(IOType.d0(0f), result[1])
        assertContentEquals(IOType.d0(0f), result[2])
    }
}
