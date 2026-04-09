package com.wsr.cpu

import com.wsr.base.data.DataBuffer
import com.wsr.base.data.IDataBufferGenerator
import com.wsr.base.data.size
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

internal fun DataBuffer.toCPUBuffer(): CPUJvmBuffer = when (this) {
    is CPUJvmBuffer -> this
    else -> CPUJvmBuffer.create(this.toFloatArray())
}

class CPUJvmBuffer(internal val byteBuffer: ByteBuffer) : DataBuffer {
    private val floatBuffer = byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer()
    override val size = floatBuffer.capacity()
    override fun toFloatArray(): FloatArray {
        val result = FloatArray(size)
        floatBuffer.move(position = 0, limit = size) { get(result) }
        return result
    }

    override fun get(i: Int): Float = floatBuffer.get(i)

    override fun set(i: Int, value: Float) {
        floatBuffer.put(i, value)
    }

    override fun copyInto(dest: DataBuffer, destIndices: IntProgression) {
        val count = minOf(size, destIndices.size)
        when {
            dest is CPUJvmBuffer && destIndices.step == 1 -> {
                val byteCount = count * Float.SIZE_BYTES
                val start = destIndices.first * Float.SIZE_BYTES
                byteBuffer.move(position = 0, limit = byteCount) {
                    val src = slice().order(ByteOrder.nativeOrder())
                    dest.byteBuffer.move(position = start, limit = start + byteCount) {
                        put(src)
                    }
                }
            }
            else -> {
                val iterator = destIndices.iterator()
                repeat(count) {
                    if (iterator.hasNext()) dest[iterator.nextInt()] = this[it]
                }
            }
        }
    }

    override fun contentEquals(other: DataBuffer): Boolean {
        if (size != other.size) return false
        return when (other) {
            is CPUJvmBuffer -> byteBuffer == other.byteBuffer
            else -> this.toFloatArray().contentEquals(other.toFloatArray())
        }
    }

    override fun toString(): String = toFloatArray().joinToString(prefix = "CPUJvmBuffer[", postfix = "]")

    companion object Companion {
        fun create(size: Int): CPUJvmBuffer {
            val buffer = ByteBuffer.allocateDirect(size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            return CPUJvmBuffer(buffer)
        }

        fun create(value: FloatArray): CPUJvmBuffer {
            val buffer = ByteBuffer
                .allocateDirect(value.size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
                .apply { asFloatBuffer().put(value) }
            return CPUJvmBuffer(buffer)
        }

        val generator = object : IDataBufferGenerator {
            override fun create(size: Int): DataBuffer = CPUJvmBuffer.create(size)

            override fun create(value: FloatArray): DataBuffer = CPUJvmBuffer.create(value)
        }
    }
}

private inline fun <B : Buffer, T> B.move(position: Int, limit: Int = limit(), block: B.() -> T): T {
    val currentPosition = position()
    val currentLimit = this.limit()
    position(position)
    limit(limit)
    try {
        return block()
    } finally {
        position(currentPosition)
        limit(currentLimit)
    }
}
