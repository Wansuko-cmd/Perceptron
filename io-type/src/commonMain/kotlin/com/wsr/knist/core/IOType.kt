package com.wsr.knist.core

import com.wsr.knist.base.data.DataBuffer
import kotlinx.serialization.Serializable

@Serializable
sealed class IOType {
    abstract val value: DataBuffer
    abstract val shape: List<Int>
    abstract val size: Int

    @Serializable
    sealed class D0 : IOType() {
        override val shape = listOf(1)
        override val size = 1

        internal fun toLocal() = Local(value)
        fun toGlobal() = Global(value)

        @Serializable
        data class Local(override val value: DataBuffer) : D0()

        @Serializable
        data class Global(override val value: DataBuffer) : D0()
    }

    @Serializable
    sealed class D1 : IOType() {
        override val shape get() = listOf(size)

        internal fun toLocal() = Local(value, size)
        fun toGlobal() = Global(value, size)

        @Serializable
        data class Local(override val value: DataBuffer, override val size: Int = value.size) : D1()

        @Serializable
        data class Global(override val value: DataBuffer, override val size: Int = value.size) : D1()
    }

    @Serializable
    sealed class D2 : IOType() {
        override val size get() = shape.reduce { acc, i -> acc * i }
        val i get() = shape[0]
        val j get() = shape[1]

        internal fun toLocal() = Local(value, shape)
        fun toGlobal() = Global(value, shape)

        @Serializable
        data class Local(override val value: DataBuffer, override val shape: List<Int>) : D2()

        @Serializable
        data class Global(override val value: DataBuffer, override val shape: List<Int>) : D2()
    }

    @Serializable
    sealed class D3 : IOType() {
        override val size get() = shape.reduce { acc, i -> acc * i }
        val i get() = shape[0]
        val j get() = shape[1]
        val k get() = shape[2]

        internal fun toLocal() = Local(value, shape)
        fun toGlobal() = Global(value, shape)

        @Serializable
        data class Local(override val value: DataBuffer, override val shape: List<Int>) : D3()

        @Serializable
        data class Global(override val value: DataBuffer, override val shape: List<Int>) : D3()
    }

    @Serializable
    sealed class D4 : IOType() {
        override val size get() = shape.reduce { acc, i -> acc * i }
        val i get() = shape[0]
        val j get() = shape[1]
        val k get() = shape[2]
        val l get() = shape[3]

        internal fun toLocal() = Local(value, shape)
        fun toGlobal() = Global(value, shape)

        @Serializable
        data class Local(override val value: DataBuffer, override val shape: List<Int>) : D4()

        @Serializable
        data class Global(override val value: DataBuffer, override val shape: List<Int>) : D4()
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
