package com.wsr.batch

import com.wsr.base.data.DataBuffer
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.create
import kotlin.jvm.JvmName

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
    first.value.copyInto(value)
    for (i in 1 until size) {
        val src = init(i).value
        val start = i * first.value.size
        src.copyInto(value, start until start + src.size)
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
        item.value.copyInto(batchValue, start until start + item.value.size)
    }
    return Batch(
        value = batchValue,
        size = batchSize,
        shape = shape,
    )
}

@JvmName("batchD0sGet")
operator fun Batch<IOType.D0>.get(i: Int): IOType.D0 {
    val index = i * step
    return IOType.d0(value[index])
}

@JvmName("batchD1sGet")
operator fun Batch<IOType.D1>.get(i: Int): IOType.D1 {
    val index = i * step
    return IOType.D1(value.slice(index until index + step))
}

@JvmName("batchD2sGet")
operator fun Batch<IOType.D2>.get(i: Int): IOType.D2 {
    val index = i * step
    return IOType.D2(shape = shape, value = value.slice(index until index + step))
}

@JvmName("batchD3sGet")
operator fun Batch<IOType.D3>.get(i: Int): IOType.D3 {
    val index = i * step
    return IOType.D3(shape = shape, value = value.slice(index until index + step))
}

@JvmName("batchD3sGet")
operator fun Batch<IOType.D4>.get(i: Int): IOType.D4 {
    val index = i * step
    return IOType.D4(shape = shape, value = value.slice(index until index + step))
}

operator fun Batch<IOType.D0>.set(i: Int, element: IOType.D0) {
    value[i] = element.value[0]
}

operator fun Batch<IOType.D1>.set(i: Int, element: IOType.D1) {
    val start = i * step
    element.value.copyInto(value, start until start + element.value.size)
}

operator fun Batch<IOType.D2>.set(i: Int, element: IOType.D2) {
    val start = i * step
    element.value.copyInto(value, start until start + element.value.size)
}

operator fun Batch<IOType.D3>.set(i: Int, element: IOType.D3) {
    val start = i * step
    element.value.copyInto(value, start until start + element.value.size)
}

operator fun Batch<IOType.D4>.set(i: Int, element: IOType.D4) {
    val start = i * step
    element.value.copyInto(value, start until start + element.value.size)
}
