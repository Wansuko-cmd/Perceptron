@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.shape

import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.batchOf
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class D4ExtTest {
    @Test
    fun `reshapeToD3=D4バッチをD3バッチに変形`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l },
            IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l + 16f },
        )
        val result = batch.reshapeToD3(4, 2, 2)
        assertEquals(2, result.size)
        assertEquals(listOf(4, 2, 2), result.shape)
        assertContentEquals(IOType.d3(4, 2, 2) { i, j, k -> i * 4f + j * 2f + k }, result[0])
        assertContentEquals(IOType.d3(4, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 16f }, result[1])
    }
}
