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

    companion object {
        inline fun launch(block: BufferScope.() -> Unit) {
            BufferScope().use { scope -> scope.block() }
        }

        inline fun launch(block: BufferScope.() -> DataBuffer): DataBuffer = BufferScope()
            .use { scope ->
                scope.block().also { scope.remove(it) }
            }

        inline fun BufferScope.launch(block: BufferScope.() -> DataBuffer): DataBuffer = BufferScope()
            .use { scope ->
                scope.block()
                    .also { scope.remove(it) }
                    .also { this.register(it) }
            }
    }
}
