@file:OptIn(ExperimentalForeignApi::class)

package com.wsr.knist.cpu

import cnames.structs.CPUBuffer
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.base.data.IDataBufferGenerator
import com.wsr.knist.cpu.rs.com_wsr_cpu_buffer_alloc
import com.wsr.knist.cpu.rs.com_wsr_cpu_buffer_get
import com.wsr.knist.cpu.rs.com_wsr_cpu_buffer_init
import com.wsr.knist.cpu.rs.com_wsr_cpu_buffer_read_all
import com.wsr.knist.cpu.rs.com_wsr_cpu_buffer_release
import com.wsr.knist.cpu.rs.com_wsr_cpu_buffer_set
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.experimental.ExperimentalNativeApi
import kotlin.native.ref.createCleaner
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

internal fun DataBuffer.toCPUBuffer(): CPUNativeBuffer = when (this) {
    is CPUNativeBuffer -> this
    else -> CPUNativeBuffer.create(this.toFloatArray())
}

@OptIn(ExperimentalAtomicApi::class)
private class CPUNativeBufferState(val ptr: CPointer<CPUBuffer>, val released: AtomicInt)

@OptIn(ExperimentalAtomicApi::class)
class CPUNativeBuffer(val buffer: CPointer<CPUBuffer>, override val size: Int) : DataBuffer {
    private val state = CPUNativeBufferState(buffer, AtomicInt(0))

    @Suppress("UNUSED")
    @OptIn(ExperimentalNativeApi::class)
    private val cleaner = createCleaner(state) { s ->
        if (s.released.compareAndExchange(0, 1) == 0) {
            com_wsr_cpu_buffer_release(s.ptr)
        }
    }

    override fun get(i: Int): Float = com_wsr_cpu_buffer_get(buffer, i)

    override fun set(i: Int, value: Float) {
        com_wsr_cpu_buffer_set(buffer, i, value)
    }

    override fun toFloatArray(): FloatArray = FloatArray(size).also { array ->
        array.usePinned { pinned ->
            com_wsr_cpu_buffer_read_all(buffer, pinned.addressOf(0))
        }
    }

    override fun toString(): String = toFloatArray().joinToString(prefix = "CPUNativeBuffer[", postfix = "]")

    override fun release() {
        if (state.released.compareAndExchange(0, 1) == 0) {
            com_wsr_cpu_buffer_release(buffer)
        }
    }

    companion object {
        fun create(size: Int) = CPUNativeBuffer(
            buffer = com_wsr_cpu_buffer_alloc(size)!!,
            size = size,
        )

        fun create(value: FloatArray): CPUNativeBuffer {
            val buffer = value.usePinned { pinned ->
                com_wsr_cpu_buffer_init(pinned.addressOf(0), value.size)!!
            }
            return CPUNativeBuffer(buffer = buffer, size = value.size)
        }

        val generator = object : IDataBufferGenerator {
            override fun create(size: Int): DataBuffer = CPUNativeBuffer.create(size)

            override fun create(value: FloatArray): DataBuffer = CPUNativeBuffer.create(value)
        }
    }
}
