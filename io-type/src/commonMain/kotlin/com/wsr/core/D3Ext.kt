package com.wsr.core

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import kotlin.jvm.JvmName

inline fun IOType.Companion.d3(i: Int, j: Int, k: Int, init: (Int, Int, Int) -> Float = { _, _, _ -> 0f }): IOType.D3 {
    val value = FloatArray(i * j * k)
    for (_i in 0 until i) {
        for (_j in 0 until j) {
            for (_k in 0 until k) {
                value[(_i * j + _j) * k + _k] = init(_i, _j, _k)
            }
        }
    }
    return IOType.d3(shape = listOf(i, j, k), value = value)
}

inline fun IOType.Companion.d3(shape: List<Int>, init: (Int, Int, Int) -> Float = { _, _, _ -> 0f }) = d3(
    i = shape[0],
    j = shape[1],
    k = shape[2],
    init = init,
)

fun IOType.Companion.d3(shape: List<Int>, value: List<Float>) = IOType.d3(
    value = value.toFloatArray(),
    shape = shape,
)

@JvmName("d3WithElements")
fun IOType.Companion.d3(vararg elements: IOType.D2): IOType.D3 {
    val size = elements.size
    val shape = elements.first().shape
    val step = shape.reduce { acc, i -> acc * i }
    val value = DataBuffer.create(size * step)
    elements.forEachIndexed { index, item ->
        val start = index * step
        Backend.copyInto(item.value, value, start until start + item.size)
    }
    return IOType.D3(shape = listOf(size, shape[0], shape[1]), value = value)
}

fun IOType.Companion.d3(shape: List<Int>, value: FloatArray) =
    IOType.D3(shape = shape, value = DataBuffer.create(value))

operator fun IOType.D3.get(i: Int, j: Int, k: Int) = value[(i * shape[1] + j) * shape[2] + k]

operator fun IOType.D3.get(i: Int, j: Int): IOType.D1 {
    val offset = (i * shape[1] + j) * shape[2]
    val result = Backend.slice(x = value, indices = offset until offset + shape[2])
    return IOType.D1(value = result)
}

operator fun IOType.D3.get(i: Int): IOType.D2 {
    val offset = i * shape[1] * shape[2]
    val result = Backend.slice(x = value, indices = offset until offset + shape[1] * shape[2])
    return IOType.D2(
        shape = listOf(shape[1], shape[2]),
        value = result,
    )
}

operator fun IOType.D3.set(i: Int, j: Int, z: Int, element: Float) {
    value[(i * shape[1] + j) * shape[2] + z] = element
}

operator fun IOType.D3.set(i: Int, j: Int, element: IOType.D1) {
    val start = (i * shape[1] + j) * shape[2]
    Backend.copyInto(
        x = element.value,
        y = value,
        indices = start until start + element.value.size,
    )
}

operator fun IOType.D3.set(i: Int, element: IOType.D2) {
    val start = i * shape[1] * shape[2]
    Backend.copyInto(
        x = element.value,
        y = value,
        indices = start until start + element.value.size,
    )
}
