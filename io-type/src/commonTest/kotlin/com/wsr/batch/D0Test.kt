@file:Suppress("NonAsciiCharacters")

package com.wsr.batch

import com.wsr.assertContentEquals
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class D0Test {
    @Test
    fun `get=D0バッチ要素取得`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        assertContentEquals(IOType.d0(1f), batch[0])
        assertContentEquals(IOType.d0(2f), batch[1])
        assertContentEquals(IOType.d0(3f), batch[2])
    }

    @Test
    fun `set=D0バッチ要素設定`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        batch[0] = IOType.d0(10f)
        batch[2] = IOType.d0(30f)
        assertContentEquals(IOType.d0(10f), batch[0])
        assertContentEquals(IOType.d0(2f), batch[1])
        assertContentEquals(IOType.d0(30f), batch[2])
    }
}
