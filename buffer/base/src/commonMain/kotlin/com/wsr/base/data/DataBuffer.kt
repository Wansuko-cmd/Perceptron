package com.wsr.base.data

import kotlinx.serialization.Serializable

@Serializable(with = DataBufferSerializable::class)
interface DataBuffer {
    val size: Int

    val indices: IntRange get() = 0 until size

    fun toFloatArray(): FloatArray

    operator fun get(i: Int): Float
    operator fun set(i: Int, value: Float)

    fun slice(indices: IntProgression): DataBuffer

    fun copyInto(destination: DataBuffer, destinationOffset: Int = 0)

    fun contentEquals(other: DataBuffer): Boolean

    companion object
}
