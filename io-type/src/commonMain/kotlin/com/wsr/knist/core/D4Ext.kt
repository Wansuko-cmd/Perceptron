package com.wsr.knist.core

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import kotlin.jvm.JvmName

inline fun IOType.Companion.d4(
    i: Int,
    j: Int,
    k: Int,
    l: Int,
    init: (Int, Int, Int, Int) -> Float = { _, _, _, _ -> 0f },
): IOType.D4 {
    val value = FloatArray(i * j * k * l)
    for (_i in 0 until i) {
        for (_j in 0 until j) {
            for (_k in 0 until k) {
                for (_l in 0 until l) {
                    value[((_i * j + _j) * k + _k) * l + _l] = init(_i, _j, _k, _l)
                }
            }
        }
    }
    return IOType.d4(shape = listOf(i, j, k, l), value = value)
}

inline fun IOType.Companion.d4(shape: List<Int>, init: (Int, Int, Int, Int) -> Float = { _, _, _, _ -> 0f }) = d4(
    i = shape[0],
    j = shape[1],
    k = shape[2],
    l = shape[3],
    init = init,
)

fun IOType.Companion.d4(shape: List<Int>, value: List<Float>) = IOType.d4(
    value = value.toFloatArray(),
    shape = shape,
)

@JvmName("d4WithElements")
fun IOType.Companion.d4(vararg elements: IOType.D3): IOType.D4 {
    val size = elements.size
    val shape = elements.first().shape
    val step = shape.reduce { acc, i -> acc * i }
    val value = DataBuffer.create(size * step)
    elements.forEachIndexed { index, item ->
        val start = index * step
        Backend.copyInto(item.value, value, start until start + item.size)
    }
    return IOType.D4(shape = listOf(size, shape[0], shape[1], shape[2]), value = value)
}

fun IOType.Companion.d4(shape: List<Int>, value: FloatArray) =
    IOType.D4(shape = shape, value = DataBuffer.create(value))

operator fun IOType.D4.get(i: Int, j: Int, k: Int, l: Int): IOType.D0 = IOType.d0(
    value[
        (
            (i * shape[1] + j) * shape[2] +
                k
            ) *
            shape[3] +
            l,
    ],
)

operator fun IOType.D4.get(i: Int, j: Int, k: Int): IOType.D1 {
    val offset = ((i * shape[1] + j) * shape[2] + k) * shape[3]
    val result = Backend.slice(x = value, indices = offset until offset + shape[3])
    return IOType.D1(value = result)
}

operator fun IOType.D4.get(i: Int, j: Int): IOType.D2 {
    val offset = (i * shape[1] + j) * shape[2] * shape[3]
    val result = Backend.slice(x = value, indices = offset until offset + shape[2] * shape[3])
    return IOType.D2(
        shape = listOf(shape[2], shape[3]),
        value = result,
    )
}

operator fun IOType.D4.get(i: Int): IOType.D3 {
    val offset = i * shape[1] * shape[2] * shape[3]
    val result = Backend.slice(x = value, indices = offset until offset + shape[1] * shape[2] * shape[3])
    return IOType.D3(
        shape = listOf(shape[1], shape[2], shape[3]),
        value = result,
    )
}

operator fun IOType.D4.set(i: Int, j: Int, k: Int, l: Int, element: Float) {
    value[((i * shape[1] + j) * shape[2] + k) * shape[3] + l] = element
}
