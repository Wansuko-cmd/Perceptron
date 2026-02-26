@file:OptIn(ExperimentalAtomicApi::class)

package com.wsr.gpu

import com.wsr.base.data.DataBuffer
import com.wsr.base.data.IDataBufferGenerator
import java.lang.ref.Cleaner
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.concurrent.atomics.minusAssign

private val allocateSize = AtomicInt(0)
private const val MAX_ALLOCATE_SIZE = 1_500_000_000

internal fun DataBuffer.toGPUBuffer(context: Long, native: JBuffer): GPUJvmBuffer = when (this) {
    is GPUJvmBuffer -> this
    else -> GPUJvmBuffer.create(this.toFloatArray(), context, native)
}

class GPUJvmBuffer(
    override val size: Int,
    internal val ptr: Long,
    private val context: Long,
    private val native: JBuffer,
) : DataBuffer {
    init {
        val size = size
        val ptr = ptr
        val context = context
        val native = native
        if (allocateSize.addAndFetch(size) >= MAX_ALLOCATE_SIZE) System.gc()
        cleaner.register(this) {
            native.release(ptr, context)
            allocateSize.minusAssign(size)
        }
    }

    override fun toFloatArray(): FloatArray = native.readAll(ptr, context)

    override fun get(i: Int): Float = native.readAll(ptr, context)[i]

    override fun set(i: Int, value: Float) {
        native.write(ptr, i, value, context)
    }

    override fun slice(indices: IntRange): DataBuffer {
        val ptr = ptr
        val context = context
        val native = native
        return GPUJvmBuffer(
            size = indices.count(),
            ptr = native.slice(ptr, indices.first, indices.last + 1, context),
            context = context,
            native = native,
        )
    }

    override fun copyInto(destination: DataBuffer, destinationOffset: Int) {
        when (destination) {
            is GPUJvmBuffer -> {
                native.copyInto(ptr, destination.ptr, destinationOffset, context)
            }
            else -> {
                val value = toFloatArray()
                for (i in indices) destination[i + destinationOffset] = value[i]
            }
        }
    }

    override fun contentEquals(other: DataBuffer): Boolean {
        if (size != other.size) return false
        return when (other) {
            is GPUJvmBuffer -> native.contentEquals(ptr, other.ptr, context)
            else -> toFloatArray().contentEquals(other.toFloatArray())
        }
    }

    override fun toString(): String = toFloatArray().joinToString(prefix = "[", postfix = "]")

    companion object {
        private val cleaner = Cleaner.create()

        fun create(size: Int, context: Long, native: JBuffer): GPUJvmBuffer {
            val ptr = native.allocate(size, context)
            return GPUJvmBuffer(size = size, ptr = ptr, context = context, native = native)
        }

        fun create(value: FloatArray, context: Long, native: JBuffer): GPUJvmBuffer {
            val ptr = native.init(value, context)
            return GPUJvmBuffer(size = value.size, ptr = ptr, context = context, native = native)
        }

        fun createGenerator(context: Long, native: JBuffer) = object : IDataBufferGenerator {
            override fun create(size: Int): DataBuffer = create(
                size = size,
                context = context,
                native = native,
            )

            override fun create(value: FloatArray): DataBuffer = create(
                value = value,
                context = context,
                native = native,
            )
        }
    }
}
