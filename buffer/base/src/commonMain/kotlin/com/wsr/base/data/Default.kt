package com.wsr.base.data

import kotlinx.serialization.Serializable

@Serializable
data class Default(private val value: FloatArray) : DataBuffer {
    constructor(size: Int) : this(value = FloatArray(size))

    override val size = value.size

    override fun toFloatArray(): FloatArray = value

    override operator fun get(i: Int): Float = value[i]
    override operator fun set(i: Int, value: Float) {
        this.value[i] = value
    }

    override fun slice(indices: IntRange): DataBuffer = Default(value.sliceArray(indices))

    override fun copyInto(destination: DataBuffer, destinationOffset: Int) {
        when (destination) {
            is Default -> value.copyInto(destination.value, destinationOffset)
            else -> for (i in indices) destination[destinationOffset + i] = this[i]
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DataBuffer) return false
        if (size != other.size) return false
        return when (other) {
            is Default -> value.contentEquals(other.value)
            else -> value.contentEquals(other.toFloatArray())
        }
    }

    override fun hashCode(): Int {
        var result = size
        result = 31 * result + value.contentHashCode()
        result = 31 * result + indices.hashCode()
        return result
    }

    companion object {
        val generator = object : IDataBufferGenerator {
            override fun create(size: Int): DataBuffer = Default(size)

            override fun create(value: FloatArray): DataBuffer = Default(value)
        }
    }
}
