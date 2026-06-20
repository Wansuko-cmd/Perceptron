package com.wsr.knist.core

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer

@PublishedApi
internal inline fun IOType.Companion.d1Impl(size: Int, init: (Int) -> Float = { 0f }): IOType.D1.Global {
    val value = FloatArray(size)
    for (i in 0 until size) value[i] = init(i)
    return d1Impl(value = value)
}

@PublishedApi
internal inline fun IOType.Companion.d1Impl(shape: List<Int>, init: (Int) -> Float = { 0f }): IOType.D1.Global =
    d1Impl(shape[0], init)

internal fun IOType.Companion.d1Impl(value: List<Float>): IOType.D1.Global = d1Impl(value = value.toFloatArray())

@PublishedApi
internal fun IOType.Companion.d1Impl(value: FloatArray): IOType.D1.Global =
    IOType.D1.Global(value = DataBuffer.create(value))

operator fun IOType.D1.get(index: Int): IOType.D0 = IOType.d0(value[index])

operator fun IOType.D1.set(index: Int, element: Float) {
    value[index] = element
}

fun IOType.D1.concat(other: IOType.D1): IOType.D1.Global {
    val newSize = size + other.size
    val result = DataBuffer.create(newSize)
    Backend.copyInto(value, result, 0 until size)
    Backend.copyInto(other.value, result, size until newSize)
    return IOType.D1.Global(value = result)
}
