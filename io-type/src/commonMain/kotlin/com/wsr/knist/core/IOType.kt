package com.wsr.knist.core

import com.wsr.knist.base.data.DataBuffer
import kotlinx.serialization.Serializable

@Serializable
sealed class IOType {
    abstract val value: DataBuffer
    abstract val shape: List<Int>
    abstract val size: Int

    @Serializable
    data class D0(override val value: DataBuffer) : IOType() {
        override val shape = listOf(1)
        override val size = 1

        override fun equals(other: Any?): Boolean = super.equals(other)

        override fun hashCode(): Int = super.hashCode()
    }

    @Serializable
    data class D1(override val value: DataBuffer, override val size: Int = value.size) : IOType() {
        override val shape = listOf(size)

        override fun equals(other: Any?): Boolean = super.equals(other)
        override fun hashCode(): Int = super.hashCode()
    }

    @Serializable
    data class D2(override val value: DataBuffer, override val shape: List<Int>) : IOType() {
        override val size = shape.reduce { acc, i -> acc * i }
        val i = shape[0]
        val j = shape[1]

        override fun equals(other: Any?): Boolean = super.equals(other)
        override fun hashCode(): Int = super.hashCode()
    }

    @Serializable
    data class D3(override val value: DataBuffer, override val shape: List<Int>) : IOType() {
        override val size = shape.reduce { acc, i -> acc * i }
        val i = shape[0]
        val j = shape[1]
        val k = shape[2]

        override fun equals(other: Any?): Boolean = super.equals(other)
        override fun hashCode(): Int = super.hashCode()
    }

    @Serializable
    data class D4(override val value: DataBuffer, override val shape: List<Int>) : IOType() {
        override val size = shape.reduce { acc, i -> acc * i }
        val i = shape[0]
        val j = shape[1]
        val k = shape[2]
        val l = shape[3]

        override fun equals(other: Any?): Boolean = super.equals(other)
        override fun hashCode(): Int = super.hashCode()
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is IOType) return false

        if (value != other.value) return false
        if (shape != other.shape) return false

        return true
    }

    override fun hashCode(): Int {
        var result = value.hashCode()
        result = 31 * result + shape.hashCode()
        return result
    }
}
