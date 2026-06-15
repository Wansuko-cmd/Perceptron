@file:OptIn(ExperimentalAtomicApi::class)

package com.wsr.knist.gpu

import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.base.data.IDataBufferGenerator
import java.lang.ref.Cleaner
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.concurrent.atomics.minusAssign

private val allocateSize = AtomicInt(0)
private const val MAX_ALLOCATE_SIZE = 1_500_000_000

internal fun DataBuffer.toGPUBuffer(runtime: Long, native: JBuffer): GPUJvmBuffer = when (this) {
    is GPUJvmBuffer -> this
    else -> GPUJvmBuffer.create(this.toFloatArray(), runtime, native)
}

class GPUJvmBuffer(
    override val size: Int,
    internal val ptr: Long,
    private val runtime: Long,
    private val native: JBuffer,
) : DataBuffer {
    private val isReleased = AtomicBoolean(false)

    init {
        val size = size
        val ptr = ptr
        val runtime = runtime
        val native = native
        val isReleased = isReleased
        if (allocateSize.addAndFetch(size) >= MAX_ALLOCATE_SIZE) System.gc()
        cleaner.register(this) {
            if (!isReleased.getAndSet(true)) {
                native.release(ptr, runtime)
                allocateSize.minusAssign(size)
            }
        }
    }

    override fun get(i: Int): Float = native.readAll(ptr, runtime)[i]

    override fun set(i: Int, value: Float) {
        native.write(ptr, i, value, runtime)
    }

    override fun toFloatArray(): FloatArray = native.readAll(ptr, runtime)

    override fun toString(): String = toFloatArray().joinToString(prefix = "GPUJvmBuffer[", postfix = "]")

    override fun release() {
        if (!isReleased.getAndSet(true)) {
            native.release(ptr, runtime)
            allocateSize.minusAssign(size)
        }
    }

    companion object {
        private val cleaner = Cleaner.create()

        fun create(size: Int, runtime: Long, native: JBuffer): GPUJvmBuffer {
            val ptr = native.allocate(size, runtime)
            return GPUJvmBuffer(size = size, ptr = ptr, runtime = runtime, native = native)
        }

        fun create(value: FloatArray, runtime: Long, native: JBuffer): GPUJvmBuffer {
            val ptr = native.init(value, runtime)
            return GPUJvmBuffer(size = value.size, ptr = ptr, runtime = runtime, native = native)
        }

        fun createGenerator(runtime: Long, native: JBuffer) = object : IDataBufferGenerator {
            override fun create(size: Int): DataBuffer = create(
                size = size,
                runtime = runtime,
                native = native,
            )

            override fun create(value: FloatArray): DataBuffer = create(
                value = value,
                runtime = runtime,
                native = native,
            )
        }
    }
}
