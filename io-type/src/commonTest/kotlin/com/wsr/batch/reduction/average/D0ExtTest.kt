@file:Suppress("NonAsciiCharacters")

package com.wsr.batch.reduction.average

import com.wsr.assertContentEquals
import com.wsr.batch.batchOf
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class D0ExtTest {
    @Test
    fun `batchAverage=D0バッチの平均`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(2f), IOType.d0(4f), IOType.d0(6f))
        val result = batch.batchAverage()
        assertContentEquals(IOType.d0(4f), result)
    }
}
