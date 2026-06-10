@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core

import com.wsr.knist.base.BufferScope
import com.wsr.knist.core.elementwise.operation.plus.plus
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test
import kotlin.test.assertTrue

class IOScopeTest {

    @Test
    fun `plus の結果が bufferScope に登録される`() = ioTypeTestRule {
        val bufferScope = BufferScope.Local()
        val ioScope = IOScope(bufferScope)
        val a = IOType.d2(2, 2) { _, _ -> 1f }
        val b = IOType.d2(2, 2) { _, _ -> 2f }
        val result = with(ioScope) { a + b }
        assertTrue(bufferScope.buffers.contains(result.value))
    }

    @Test
    fun `scope外ではbufferScopeに登録されない`() = ioTypeTestRule {
        val bufferScope = BufferScope.Local()
        val a = IOType.d2(2, 2) { _, _ -> 1f }
        val b = IOType.d2(2, 2) { _, _ -> 2f }
        val result = a + b
        assertTrue(!bufferScope.buffers.contains(result.value))
    }
}
