@file:Suppress("NonAsciiCharacters")

package com.wsr.batch

import com.wsr.assertContentEquals
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.ioTypeTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class D1Test {
    @Test
    fun `i=D1バッチのi次元`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        assertEquals(3, batch.i)
    }

    @Test
    fun `get=D1バッチ要素取得`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        assertContentEquals(IOType.d1(listOf(1f, 2f, 3f)), batch[0])
        assertContentEquals(IOType.d1(listOf(4f, 5f, 6f)), batch[1])
    }

    @Test
    fun `set=D1バッチ要素設定`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        batch[0] = IOType.d1(listOf(7f, 8f, 9f))
        assertContentEquals(IOType.d1(listOf(7f, 8f, 9f)), batch[0])
        assertContentEquals(IOType.d1(listOf(4f, 5f, 6f)), batch[1])
    }
}
