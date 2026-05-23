package com.wsr.core

import com.wsr.base.data.DataBuffer
import kotlin.jvm.JvmName

inline fun IOType.Companion.d1(size: Int, init: (Int) -> Float = { 0f }): IOType.D1 {
    val value = FloatArray(size)
    for (i in 0 until size) value[i] = init(i)
    return IOType.d1(value = value)
}

inline fun IOType.Companion.d1(shape: List<Int>, init: (Int) -> Float = { 0f }) = d1(shape[0], init)

fun IOType.Companion.d1(value: List<Float>) = IOType.d1(value = value.toFloatArray())

@JvmName("d1WithElements")
fun IOType.Companion.d1(vararg elements: Float) = IOType.d1(value = elements)

fun IOType.Companion.d1(value: FloatArray) = IOType.D1(value = DataBuffer.create(value))

operator fun IOType.D1.get(index: Int) = value[index]

operator fun IOType.D1.set(index: Int, element: Float) {
    value[index] = element
}
