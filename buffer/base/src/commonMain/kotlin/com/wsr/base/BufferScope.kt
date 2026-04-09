package com.wsr.base

import com.wsr.base.data.DataBuffer

sealed interface BufferScope {
    fun register(buffer: DataBuffer)

    fun remove(buffer: DataBuffer)

    fun close()

    object Global : BufferScope {
        override fun register(buffer: DataBuffer) {
        }

        override fun remove(buffer: DataBuffer) {
        }

        override fun close() {
        }
    }

    class Local(val buffers: ArrayList<DataBuffer> = arrayListOf()) : BufferScope, AutoCloseable {
        override fun register(buffer: DataBuffer) {
            buffers.add(buffer)
        }

        override fun remove(buffer: DataBuffer) {
            buffers.add(buffer)
        }

        override fun close() {
            for (i in buffers.indices) buffers[i].release()
        }
    }

    companion object {
        inline fun launch(block: BufferScope.() -> Unit) {
            Local().use { scope -> scope.block() }
        }

        inline fun launch(block: BufferScope.() -> DataBuffer): DataBuffer = Local()
            .use { scope ->
                scope.block().also { scope.remove(it) }
            }

        inline fun BufferScope.launch(block: BufferScope.() -> DataBuffer): DataBuffer = Local()
            .use { scope ->
                scope.block()
                    .also { scope.remove(it) }
                    .also { this.register(it) }
            }
    }
}
