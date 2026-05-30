@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.shape

import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class CommonExtTest {
    @Test
    fun `toBatch=D0リストからバッチ作成`() = ioTypeTestRule {
        val list = listOf(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val batch = list.toBatch()
        assertEquals(3, batch.size)
        assertContentEquals(IOType.d0(1f), batch[0])
        assertContentEquals(IOType.d0(2f), batch[1])
        assertContentEquals(IOType.d0(3f), batch[2])
    }

    @Test
    fun `toBatch=D1リストからバッチ作成`() = ioTypeTestRule {
        val list = listOf(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val batch = list.toBatch()
        assertEquals(2, batch.size)
        assertContentEquals(IOType.d1(listOf(1f, 2f, 3f)), batch[0])
        assertContentEquals(IOType.d1(listOf(4f, 5f, 6f)), batch[1])
    }
}
