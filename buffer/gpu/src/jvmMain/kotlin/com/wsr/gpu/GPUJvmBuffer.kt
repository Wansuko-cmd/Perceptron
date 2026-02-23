package com.wsr.gpu

import com.wsr.base.data.DataBuffer
import com.wsr.base.data.Default
import com.wsr.base.data.IDataBufferGenerator
import java.lang.ref.Cleaner

internal fun DataBuffer.toGPUBuffer(context: Long, native: JBuffer): GPUJvmBuffer = when (this) {
    is GPUJvmBuffer -> this
    else -> GPUJvmBuffer.create(this.toFloatArray(), context, native)
}

data class GPUJvmBuffer(
    override val size: Int,
    internal val ptr: Long,
    private val context: Long,
    private val native: JBuffer,
) : DataBuffer {
    init {
        cleaner.register(this) { native.release(ptr, context) }
    }

    override fun toFloatArray(): FloatArray = native.readAll(ptr, context)

    override fun get(i: Int): Float = native.readAll(ptr, context)[i]

    override fun set(i: Int, value: Float) {
        native.write(ptr, i, value, context)
    }

    override fun slice(indices: IntRange): DataBuffer {
        val slice = toFloatArray().sliceArray(indices)
        return Default.generator.create(slice)
    }

    override fun copyInto(destination: DataBuffer, destinationOffset: Int) {
        val value = toFloatArray()
        for (i in indices) destination[i + destinationOffset] = value[i]
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DataBuffer) return false
        return size == other.size && toFloatArray().contentEquals(other.toFloatArray())
    }

    override fun hashCode(): Int {
        var result = size
        result = 31 * result + this.toFloatArray().contentHashCode()
        return result
    }

    override fun toString(): String = toFloatArray().joinToString()

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
