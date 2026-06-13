@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.reduction.average
import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d4
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class D4ExtTest {
    @Test
    fun `batchAverage=D4バッチのバッチ平均`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l },
            IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l + 16f },
        )
        val result = batch.batchAverage()
        assertContentEquals(
            IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l + 8f },
            result,
        )
    }
}
