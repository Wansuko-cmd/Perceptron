@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.reduction.average
import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class D0ExtTest {
    @Test
    fun `batchAverage=D0バッチの平均`() = ioTypeTestRule {
        val batch = Batch.of(IOType.d0(2f), IOType.d0(4f), IOType.d0(6f))
        val result = batch.batchAverage()
        assertContentEquals(IOType.d0(4f), result)
    }
}
