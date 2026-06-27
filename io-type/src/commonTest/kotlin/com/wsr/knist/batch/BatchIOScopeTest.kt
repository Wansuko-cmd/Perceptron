@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch

import com.wsr.knist.BufferScope
import com.wsr.knist.assertContentEquals
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.launch
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class BatchIOScopeTest {

    @Test
    fun `Batch factory がscope内でLocal型を返し bufferScope に登録される`() = ioTypeTestRule {
        val bufferScope = BufferScope()
        val ioScope = IOScope(bufferScope)
        val result: Batch<IOType.D0.Local> = with(ioScope) { Batch.d0(3) { it.toFloat() } }
        assertTrue(bufferScope.buffers.contains(result.value))
    }

    @Test
    fun `Batch factory がscope外でGlobal型を返し bufferScope に登録されない`() = ioTypeTestRule {
        val batch: Batch<IOType.D0.Global> = Batch.d0(3) { it.toFloat() }
        val bufferScope = BufferScope()
        assertFalse(bufferScope.buffers.contains(batch.value))
    }

    @Test
    fun `ScopeOp の結果がLocal型になり bufferScope に登録される`() = ioTypeTestRule {
        val bufferScope = BufferScope()
        val ioScope = IOScope(bufferScope)
        val batch = Batch.of(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val result: Batch<IOType.D0.Local> = with(ioScope) { batch + 1f }
        assertTrue(bufferScope.buffers.contains(result.value))
    }

    @Test
    fun `ScopeOp 連鎖でも全中間バッファが bufferScope に登録される`() = ioTypeTestRule {
        val bufferScope = BufferScope()
        val ioScope = IOScope(bufferScope)
        val batch = Batch.of(IOType.d0(1f), IOType.d0(2f))
        val step1: Batch<IOType.D0.Local> = with(ioScope) { batch + 1f }
        val step2: Batch<IOType.D0.Local> = with(ioScope) { step1 * 2f }
        assertTrue(bufferScope.buffers.contains(step1.value))
        assertTrue(bufferScope.buffers.contains(step2.value))
    }

    @Test
    fun `launch内のBatch中間バッファはscope終了後にreleaseされる`() = ioTypeTestRule {
        val intermediate = TrackingDataBuffer(1)
        IOScope.launch {
            bufferScope.register(intermediate)
            IOType.d0(0f)
        }
        assertTrue(intermediate.released)
    }

    @Test
    fun `ScopeOp の計算結果が正しい`() = ioTypeTestRule {
        val bufferScope = BufferScope()
        val ioScope = IOScope(bufferScope)
        val batch = Batch.of(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val result: Batch<IOType.D0.Local> = with(ioScope) { batch + 10f }
        assertContentEquals(IOType.d0(11f), result[0])
        assertContentEquals(IOType.d0(12f), result[1])
        assertContentEquals(IOType.d0(13f), result[2])
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
