package com.wsr.knist.base

import com.wsr.knist.base.data.DataBuffer

sealed interface BufferScope : AutoCloseable {
    fun register(buffer: DataBuffer)

    fun remove(buffer: DataBuffer)

    override fun close()

    object Global : BufferScope {
        override fun register(buffer: DataBuffer) {
        }

        override fun remove(buffer: DataBuffer) {
        }

        override fun close() {
        }
    }

    class Local(val buffers: MutableSet<DataBuffer> = mutableSetOf()) : BufferScope {
        override fun register(buffer: DataBuffer) {
            buffers.add(buffer)
        }

        override fun remove(buffer: DataBuffer) {
            buffers.remove(buffer)
        }

        override fun close() {
            buffers.forEach { it.release() }
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
