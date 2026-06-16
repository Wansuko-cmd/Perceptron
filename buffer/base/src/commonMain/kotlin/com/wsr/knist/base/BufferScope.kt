package com.wsr.knist.base

import com.wsr.knist.base.data.DataBuffer

class BufferScope(val buffers: MutableSet<DataBuffer> = mutableSetOf()) : AutoCloseable {
    fun register(buffer: DataBuffer) {
        buffers.add(buffer)
    }

    fun remove(buffer: DataBuffer) {
        buffers.remove(buffer)
    }

    override fun close() {
        buffers.forEach { it.release() }
    }
}
