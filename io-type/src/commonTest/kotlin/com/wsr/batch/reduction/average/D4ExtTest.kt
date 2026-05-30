@file:Suppress("NonAsciiCharacters")

package com.wsr.batch.reduction.average

import com.wsr.assertContentEquals
import com.wsr.batch.batchOf
import com.wsr.core.IOType
import com.wsr.core.d4
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class D4ExtTest {
    @Test
    fun `batchAverage=D4バッチのバッチ平均`() = ioTypeTestRule {
        val batch = batchOf(
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
