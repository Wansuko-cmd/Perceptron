@file:OptIn(ExperimentalForeignApi::class)

package com.wsr.cpu

import com.wsr.base.data.DataBuffer
import com.wsr.base.data.IDataBufferGenerator
import kotlin.experimental.ExperimentalNativeApi
import kotlin.native.ref.createCleaner
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.FloatVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.free
import kotlinx.cinterop.get
import kotlinx.cinterop.nativeHeap
import kotlinx.cinterop.plus
import kotlinx.cinterop.set
import kotlinx.cinterop.usePinned
import platform.posix.memcmp
import platform.posix.memcpy

internal fun DataBuffer.toCPUBuffer(): CPUNativeBuffer = when (this) {
    is CPUNativeBuffer -> this
    else -> CPUNativeBuffer.create(this.toFloatArray())
}

class CPUNativeBuffer(val buffer: CPointer<FloatVar>, override val size: Int) : DataBuffer {
    @Suppress("UNUSED")
    @OptIn(ExperimentalNativeApi::class)
    private val cleaner = createCleaner(buffer) { ptr ->
        nativeHeap.free(ptr)
    }

    override fun toFloatArray(): FloatArray = FloatArray(size).also { array ->
        array.usePinned { pinned ->
            val byteSize = (size * Float.SIZE_BYTES).toULong()
            memcpy(pinned.addressOf(0), buffer, byteSize)
        }
    }

    override fun get(i: Int): Float = buffer[i]

    override fun set(i: Int, value: Float) {
        buffer[i] = value
    }

    override fun slice(indices: IntRange): DataBuffer {
        val size = indices.last - indices.first + 1
        val buffer = nativeHeap.allocArray<FloatVar>(size)
        val byteSize = (size * Float.SIZE_BYTES).toULong()
        memcpy(buffer, this.buffer + indices.first, byteSize)
        return CPUNativeBuffer(buffer = buffer, size = size)
    }

    override fun copyInto(destination: DataBuffer, destinationOffset: Int) {
        when (destination) {
            is CPUNativeBuffer -> {
                val byteSize = (size * Float.SIZE_BYTES).toULong()
                memcpy(destination.buffer + destinationOffset, buffer, byteSize)
            }

            else -> for (i in indices) destination[i + destinationOffset] = this[i]
        }
    }

    override fun contentEquals(other: DataBuffer): Boolean {
        if (size != other.size) return false
        return when (other) {
            is CPUNativeBuffer -> {
                val byteSize = (size * Float.SIZE_BYTES).toULong()
                memcmp(buffer, other.buffer, byteSize) == 0
            }
            else -> this.toFloatArray().contentEquals(other.toFloatArray())
        }
    }

    override fun toString(): String = toFloatArray().joinToString(prefix = "[", postfix = "]")

    companion object {
        fun create(size: Int) = CPUNativeBuffer(
            buffer = nativeHeap.allocArray(size),
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
