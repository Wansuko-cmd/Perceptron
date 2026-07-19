package com.wsr.knist.cpu

import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.base.data.IDataBufferGenerator
import java.lang.ref.Cleaner
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

private val allocatedBytes = AtomicLong(0)
private const val MAX_ALLOCATED_BYTES = 1_500_000_000L

internal fun DataBuffer.toCPUBuffer(): CPUJvmBuffer = when (this) {
    is CPUJvmBuffer -> this
    else -> CPUJvmBuffer.create(this.toFloatArray())
}

class CPUJvmBuffer private constructor(
    internal val address: Long,
    override val size: Int,
) : DataBuffer {
    private val isReleased = AtomicBoolean(false)

    init {
        val address = address
        val byteSize = size * Float.SIZE_BYTES
        val isReleased = isReleased
        if (allocatedBytes.addAndGet(byteSize.toLong()) >= MAX_ALLOCATED_BYTES) System.gc()
        cleaner.register(this) {
            if (!isReleased.getAndSet(true)) {
                JBuffer.release(address)
                allocatedBytes.addAndGet(-byteSize.toLong())
            }
        }
    }

    override fun get(i: Int): Float = JBuffer.get(address, i)

    override fun set(i: Int, value: Float) {
        JBuffer.set(address, i, value)
    }

    override fun toFloatArray(): FloatArray = JBuffer.readAll(address)

    override fun toString(): String = toFloatArray().joinToString(prefix = "CPUJvmBuffer[", postfix = "]")

    override fun release() {
        if (!isReleased.getAndSet(true)) {
            JBuffer.release(address)
            allocatedBytes.addAndGet(-(size * Float.SIZE_BYTES).toLong())
        }
    }

    companion object Companion {
        private val cleaner = Cleaner.create()

        fun create(size: Int): CPUJvmBuffer {
            val address = JBuffer.alloc(size)
            return CPUJvmBuffer(address, size)
        }

        fun create(value: FloatArray): CPUJvmBuffer {
            val address = JBuffer.alloc(value.size)
            JBuffer.writeAll(address, value)
            return CPUJvmBuffer(address, value.size)
        }

        val generator = object : IDataBufferGenerator {
            override fun create(size: Int): DataBuffer = CPUJvmBuffer.create(size)

            override fun create(value: FloatArray): DataBuffer = CPUJvmBuffer.create(value)
        }
    }
}
