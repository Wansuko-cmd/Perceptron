package com.wsr.batch

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.indices
import com.wsr.core.IOType

class Batch<out T : IOType>(val value: DataBuffer, val size: Int, val shape: List<Int>) {
    val step = shape.reduce { acc, i -> acc * i }
    val indices = 0 until size

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Batch<*>) return false

        if (size != other.size) return false
        if (value != other.value) return false
        if (shape != other.shape) return false

        return true
    }

    override fun hashCode(): Int {
        var result = size
        result = 31 * result + value.hashCode()
        result = 31 * result + shape.hashCode()
        return result
    }

    override fun toString(): String = "Batch(shape=$shape, size=$size, value=$value)"
}

inline fun <T : IOType> Batch(size: Int, init: (index: Int) -> T): Batch<T> {
    val first = init(0)
    val value = DataBuffer.create(size * first.value.size)
    Backend.copyInto(first.value, value, first.value.indices)
    for (i in 1 until size) {
        val src = init(i).value
        val start = i * first.value.size
        Backend.copyInto(src, value, start until start + src.size)
    }
    return Batch(
        value = value,
        size = size,
        shape = first.shape,
    )
}

fun <T : IOType> batchOf(vararg elements: T): Batch<T> {
    val batchSize = elements.size
    val shape = elements.first().shape
    val step = shape.reduce { acc, i -> acc * i }
    val batchValue = DataBuffer.create(batchSize * step)
    elements.forEachIndexed { index, item ->
        val start = index * step
        Backend.copyInto(item.value, batchValue, start until start + item.value.size)
    }
    return Batch(
        value = batchValue,
        size = batchSize,
        shape = shape,
    )
}
