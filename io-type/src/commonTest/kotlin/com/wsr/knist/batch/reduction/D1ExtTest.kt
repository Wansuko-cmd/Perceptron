@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.reduction

import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.batchOf
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class D1ExtTest {
    @Test
    fun `sum=D1バッチの各要素の合計`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = batch.sum()
        assertContentEquals(IOType.d0(6f), result[0])
        assertContentEquals(IOType.d0(15f), result[1])
    }

    @Test
    fun `max=D1バッチの各要素の最大値`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 3f, 2f)),
            IOType.d1(listOf(6f, 4f, 5f)),
        )
        val result = batch.max()
        assertContentEquals(IOType.d0(3f), result[0])
        assertContentEquals(IOType.d0(6f), result[1])
    }

    @Test
    fun `min=D1バッチの各要素の最小値`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(3f, 1f, 2f)),
            IOType.d1(listOf(5f, 6f, 4f)),
        )
        val result = batch.min()
        assertContentEquals(IOType.d0(1f), result[0])
        assertContentEquals(IOType.d0(4f), result[1])
    }
}
