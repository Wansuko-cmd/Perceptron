@file:OptIn(ExperimentalForeignApi::class)

package com.wsr.knist.cpu

import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.base.data.IDataBufferGenerator
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.experimental.ExperimentalNativeApi
import kotlin.native.ref.createCleaner
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.FloatVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.get
import kotlinx.cinterop.nativeHeap
import kotlinx.cinterop.set
import kotlinx.cinterop.usePinned
import platform.posix.memcpy

internal fun DataBuffer.toCPUBuffer(): CPUNativeBuffer = when (this) {
    is CPUNativeBuffer -> this
    else -> CPUNativeBuffer.create(this.toFloatArray())
}

@OptIn(ExperimentalAtomicApi::class)
private class CPUNativeBufferState(val ptr: CPointer<FloatVar>, val released: AtomicInt)

@OptIn(ExperimentalAtomicApi::class)
class CPUNativeBuffer(val buffer: CPointer<FloatVar>, override val size: Int) : DataBuffer {
    private val state = CPUNativeBufferState(buffer, AtomicInt(0))

    @Suppress("UNUSED")
    @OptIn(ExperimentalNativeApi::class)
    private val cleaner = createCleaner(state) { s ->
        if (s.released.compareAndExchange(0, 1) == 0) {
            nativeHeap.free(s.ptr.rawValue)
        }
    }

    override fun get(i: Int): Float = buffer[i]

    override fun set(i: Int, value: Float) {
        buffer[i] = value
    }

    override fun toFloatArray(): FloatArray = FloatArray(size).also { array ->
        array.usePinned { pinned ->
            val byteSize = (size * Float.SIZE_BYTES).toULong()
            memcpy(pinned.addressOf(0), buffer, byteSize)
        }
    }

    override fun toString(): String = toFloatArray().joinToString(prefix = "CPUNativeBuffer[", postfix = "]")

    override fun release() {
        if (state.released.compareAndExchange(0, 1) == 0) {
            nativeHeap.free(buffer.rawValue)
        }
    }

    companion object {
        fun create(size: Int) = CPUNativeBuffer(
            buffer = nativeHeap.allocArray<FloatVar>(size),
            size = size,
        )

        fun create(value: FloatArray): CPUNativeBuffer {
            val buffer = nativeHeap.allocArray<FloatVar>(value.size)
            value.usePinned { pinned ->
                val size = (value.size * Float.SIZE_BYTES).toULong()
                memcpy(buffer, pinned.addressOf(0), size)
            }
            return CPUNativeBuffer(buffer = buffer, size = value.size)
        }

        val generator = object : IDataBufferGenerator {
            override fun create(size: Int): DataBuffer = CPUNativeBuffer.create(size)

            override fun create(value: FloatArray): DataBuffer = CPUNativeBuffer.create(value)
        }
    }
}
