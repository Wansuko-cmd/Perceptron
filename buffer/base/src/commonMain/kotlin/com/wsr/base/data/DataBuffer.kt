package com.wsr.base.data

import kotlinx.serialization.Serializable

@Serializable(with = DataBufferSerializable::class)
interface DataBuffer {
    val size: Int
    operator fun get(i: Int): Float
    operator fun set(i: Int, value: Float)

    fun toFloatArray(): FloatArray
    fun release()

    companion object
}

val DataBuffer.indices: IntRange get() = 0 until size

fun DataBuffer.contentEquals(other: DataBuffer) = when {
    this.size != other.size -> false
    else -> this.toFloatArray().contentEquals(other.toFloatArray())
}

val IntProgression.size get() = (last - first) / step + 1
