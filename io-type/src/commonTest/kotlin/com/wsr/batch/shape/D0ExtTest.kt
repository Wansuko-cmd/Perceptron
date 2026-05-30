@file:Suppress("NonAsciiCharacters")

package com.wsr.batch.shape

import com.wsr.assertContentEquals
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.core.d1
import com.wsr.ioTypeTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class D0ExtTest {
    @Test
    fun `D1toBatch=1次元をD0バッチに変換`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 2f, 3f))
        val batch = d1.toBatch()
        assertEquals(3, batch.size)
        assertContentEquals(IOType.d0(1f), batch[0])
        assertContentEquals(IOType.d0(2f), batch[1])
        assertContentEquals(IOType.d0(3f), batch[2])
    }

    @Test
    fun `toList=D1バッチをリストに変換`() = ioTypeTestRule {
        val batch = batchOf(
            IOType.d1(listOf(1f, 2f)),
            IOType.d1(listOf(3f, 4f)),
            IOType.d1(listOf(5f, 6f)),
        )
        val list = batch.toList()
        assertEquals(3, list.size)
        assertContentEquals(IOType.d1(listOf(1f, 2f)), list[0])
        assertContentEquals(IOType.d1(listOf(3f, 4f)), list[1])
        assertContentEquals(IOType.d1(listOf(5f, 6f)), list[2])
    }
}
