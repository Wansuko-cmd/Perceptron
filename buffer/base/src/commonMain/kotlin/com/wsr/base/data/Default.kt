package com.wsr.base.data

import kotlinx.serialization.Serializable

@Serializable
class Default(private val value: FloatArray) : DataBuffer {
    constructor(size: Int) : this(value = FloatArray(size))

    override val size = value.size

    override fun toFloatArray(): FloatArray = value

    override operator fun get(i: Int): Float = value[i]
    override operator fun set(i: Int, value: Float) {
        this.value[i] = value
    }

    override fun slice(indices: IntProgression): DataBuffer {
        val result = FloatArray(indices.size)
        var targetIndex = 0
        for (sourceIndex in indices) {
            result[targetIndex++] = this[sourceIndex]
        }
        return Default(result)
    }

    override fun copyInto(dest: DataBuffer, destIndices: IntProgression) {
        val count = minOf(size, destIndices.size)
        when {
            dest is Default && destIndices.step == 1 -> {
                value.copyInto(
                    destination = dest.value,
                    destinationOffset = destIndices.first,
                    startIndex = 0,
                    endIndex = count,
                )
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
            is Default -> value.contentEquals(other.value)
            else -> value.contentEquals(other.toFloatArray())
        }
    }

    override fun toString(): String = toFloatArray().joinToString(prefix = "Default[", postfix = "]")

    companion object {
        val generator = object : IDataBufferGenerator {
            override fun create(size: Int): DataBuffer = Default(size)

            override fun create(value: FloatArray): DataBuffer = Default(value)
        }
    }
}

val IntProgression.size get() = (last - first) / step + 1
