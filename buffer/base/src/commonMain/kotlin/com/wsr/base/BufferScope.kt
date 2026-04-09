package com.wsr.base

import com.wsr.base.data.DataBuffer

class BufferScope(val buffers: ArrayList<DataBuffer> = arrayListOf()) : AutoCloseable {
    fun register(buffer: DataBuffer) {
        buffers.add(buffer)
    }

    fun remove(buffer: DataBuffer) {
        buffers.add(buffer)
    }

    override fun close() {
        for (i in buffers.indices) buffers[i].release()
    }
}
