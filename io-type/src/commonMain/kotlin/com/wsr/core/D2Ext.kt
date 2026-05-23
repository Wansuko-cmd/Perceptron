package com.wsr.core

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import kotlin.jvm.JvmName

inline fun IOType.Companion.d2(i: Int, j: Int, init: (Int, Int) -> Float = { _, _ -> 0f }): IOType.D2 {
    val value = FloatArray(i * j)
    for (_i in 0 until i) {
        for (_j in 0 until j) {
            value[_i * j + _j] = init(_i, _j)
        }
    }
    return IOType.d2(shape = listOf(i, j), value = value)
}

inline fun IOType.Companion.d2(shape: List<Int>, init: (Int, Int) -> Float = { _, _ -> 0f }) = d2(
    i = shape[0],
    j = shape[1],
    init = init,
)

fun IOType.Companion.d2(shape: List<Int>, value: List<Float>) = IOType.d2(
    value = value.toFloatArray(),
    shape = shape,
)

@JvmName("d2WithElements")
fun IOType.Companion.d2(vararg elements: IOType.D1): IOType.D2 {
    val size = elements.size
    val shape = elements.first().shape
    val step = shape.reduce { acc, i -> acc * i }
    val value = DataBuffer.create(size * step)
    elements.forEachIndexed { index, item ->
        val start = index * step
        Backend.copyInto(item.value, value, start until start + item.size)
    }
    return IOType.D2(shape = listOf(size, shape[0]), value = value)
}

fun IOType.Companion.d2(shape: List<Int>, value: FloatArray) =
    IOType.D2(shape = shape, value = DataBuffer.create(value))

operator fun IOType.D2.get(i: Int, j: Int) = value[i * shape[1] + j]

operator fun IOType.D2.get(i: Int): IOType.D1 {
    val offset = i * shape[1]
    val result = Backend.slice(x = value, indices = offset until offset + shape[1])
    return IOType.D1(result)
}

operator fun IOType.D2.set(i: Int, j: Int, element: Float) {
    value[i * shape[1] + j] = element
}

operator fun IOType.D2.set(i: Int, element: IOType.D1) {
    val start = i * shape[1]
    Backend.copyInto(
        x = element.value,
        y = value,
        indices = start until start + element.value.size,
    )
}
