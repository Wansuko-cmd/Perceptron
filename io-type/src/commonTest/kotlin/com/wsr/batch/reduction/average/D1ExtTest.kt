@file:Suppress("NonAsciiCharacters")

package com.wsr.batch.reduction.average

import com.wsr.assertContentEquals
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.core.d1
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class D1ExtTest {
    @Test
    fun `average=D1バッチの各要素の平均`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = batch.average()
        assertContentEquals(IOType.d0(2f), result[0])
        assertContentEquals(IOType.d0(5f), result[1])
    }

    @Test
    fun `batchAverage=D1バッチのバッチ平均`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = batch.batchAverage()
        assertContentEquals(IOType.d1(listOf(2.5f, 3.5f, 4.5f)), result)
    }
}
