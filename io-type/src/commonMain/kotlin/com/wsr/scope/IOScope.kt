package com.wsr.scope

import com.wsr.BufferScope
import com.wsr.base.data.DataBuffer
import com.wsr.core.IOType

class IOScope(private val scope: BufferScope = BufferScope()) : AutoCloseable {
    fun register(buffer: DataBuffer) {
        scope.register(buffer)
    }

    fun remove(buffer: DataBuffer) {
        scope.remove(buffer)
    }

    override fun close() {
        scope.close()
    }

    companion object {
        inline fun launch(block: IOScope.() -> Unit) {
            IOScope().use { scope -> scope.block() }
        }

        inline fun <T : IOType> launch(block: IOScope.() -> T): T = IOScope()
            .use { scope ->
                scope.block().also { scope.remove(it.value) }
            }

        inline fun <T: IOType> IOScope.launch(block: IOScope.() -> T): T = IOScope()
            .use { scope ->
                scope.block()
                    .also { scope.remove(it.value) }
                    .also { this.register(it.value) }
            }
    }
}
