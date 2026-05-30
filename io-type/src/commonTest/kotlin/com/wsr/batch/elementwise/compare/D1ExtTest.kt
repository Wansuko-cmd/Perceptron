@file:Suppress("NonAsciiCharacters")

package com.wsr.batch.elementwise.compare

import com.wsr.assertContentEquals
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class D1ExtTest {
    @Test
    fun `eq_Float=D1バッチとFloat値の等値比較`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(2f, 2f, 2f)),
        )
        val result = batch eq 2f
        assertContentEquals(IOType.d1(listOf(0f, 1f, 0f)), result[0])
        assertContentEquals(IOType.d1(listOf(1f, 1f, 1f)), result[1])
    }

    @Test
    fun `eq_D1s=D1バッチ同士の等値比較`() = ioTypeTestRule {
        val batch1 = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val batch2 = batchOf(
            IOType.d1(listOf(1f, 0f, 3f)),
            IOType.d1(listOf(4f, 5f, 0f)),
        )
        val result = batch1 eq batch2
        assertContentEquals(IOType.d1(listOf(1f, 0f, 1f)), result[0])
        assertContentEquals(IOType.d1(listOf(1f, 1f, 0f)), result[1])
    }

    @Test
    fun `gt_Float=D1バッチとFloat値の大比較`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(0f, 2f, 4f)),
        )
        val result = batch gt 2f
        assertContentEquals(IOType.d1(listOf(0f, 0f, 1f)), result[0])
        assertContentEquals(IOType.d1(listOf(0f, 0f, 1f)), result[1])
    }

    @Test
    fun `lt_Float=D1バッチとFloat値の小比較`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(0f, 2f, 4f)),
        )
        val result = batch lt 2f
        assertContentEquals(IOType.d1(listOf(1f, 0f, 0f)), result[0])
        assertContentEquals(IOType.d1(listOf(1f, 0f, 0f)), result[1])
    }
}
