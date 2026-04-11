package com.wsr

import com.wsr.base.data.DataBuffer

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
