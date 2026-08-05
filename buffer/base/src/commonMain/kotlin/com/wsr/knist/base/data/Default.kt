package com.wsr.knist.base.data

import kotlinx.serialization.Serializable

@Serializable
class Default(private val value: FloatArray) : DataBuffer {
    constructor(size: Int) : this(value = FloatArray(size))

    override val size = value.size

    override operator fun get(i: Int): Float = value[i]
    override operator fun set(i: Int, value: Float) {
        this.value[i] = value
    }

    override fun toFloatArray(): FloatArray = value.clone()

    override fun toString(): String = toFloatArray().joinToString(prefix = "Default[", postfix = "]")

    override fun release() {
        // do nothing
    }

    companion object {
        val generator = object : IDataBufferGenerator {
            override fun create(size: Int): DataBuffer = Default(size)

            override fun create(value: FloatArray): DataBuffer = Default(value)
        }
    }
}
