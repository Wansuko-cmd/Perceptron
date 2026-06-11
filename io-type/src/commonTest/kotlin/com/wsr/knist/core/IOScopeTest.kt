@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core

import com.wsr.knist.base.BufferScope
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.core.elementwise.operation.plus.plus
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test
import kotlin.test.assertFails
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class IOScopeTest {

    @Test
    fun `plus の結果が bufferScope に登録される`() = ioTypeTestRule {
        val bufferScope = BufferScope()
        val ioScope = IOScope(bufferScope)
        val a = IOType.d2(2, 2) { _, _ -> 1f }
        val b = IOType.d2(2, 2) { _, _ -> 2f }
        val result = with(ioScope) { a + b }
        assertTrue(bufferScope.buffers.contains(result.value))
    }

    @Test
    fun `scope外ではbufferScopeに登録されない`() = ioTypeTestRule {
        val bufferScope = BufferScope()
        val a = IOType.d2(2, 2) { _, _ -> 1f }
        val b = IOType.d2(2, 2) { _, _ -> 2f }
        val result = a + b
        assertTrue(!bufferScope.buffers.contains(result.value))
    }

    @Test
    fun `launch内の中間バッファはscope終了後にreleaseされる`() = ioTypeTestRule {
        val intermediate = TrackingDataBuffer(4)
        IOScope.launch {
            bufferScope.register(intermediate)
            val result = TrackingDataBuffer(4)
            bufferScope.register(result)
            IOType.D1(result, 4)
        }
        assertTrue(intermediate.released)
    }

    @Test
    fun `launchの戻り値のバッファはscope終了後もreleaseされない`() = ioTypeTestRule {
        val result = TrackingDataBuffer(4)
        IOScope.launch {
            bufferScope.register(result)
            IOType.D1(result, 4)
        }
        assertFalse(result.released)
    }

    @Test
    fun `launch内で例外が発生しても中間バッファはreleaseされる`() = ioTypeTestRule {
        val intermediate = TrackingDataBuffer(4)
        assertFails {
            IOScope.launch {
                bufferScope.register(intermediate)
                throw RuntimeException("test")
            }
        }
        assertTrue(intermediate.released)
    }
}

private class TrackingDataBuffer(override val size: Int) : DataBuffer {
    var released = false
    override fun get(i: Int) = 0f
    override fun set(i: Int, value: Float) {}
    override fun toFloatArray() = FloatArray(size)
    override fun release() {
        released = true
    }
}
