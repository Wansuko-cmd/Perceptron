package com.wsr.core.reshape.interleave

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.core.IOType

fun IOType.D1.interleave(other: IOType.D1): IOType.D1 {
    check(size == other.size)

    val result = DataBuffer.create(size + other.size)
    Backend.copyInto(x = value, y = result, indices = 0 until size * 2 step 2)
    Backend.copyInto(x = other.value, y = result, indices = 1 until size * 2 step 2)

    return IOType.D1(result)
}

fun IOType.D2.interleave(other: IOType.D2, axis: Int): IOType.D2 {
    check(i == other.i && j == other.j)
    val result = DataBuffer.create(size + other.size)
    val newShape = when (axis) {
        0 -> listOf(i * 2, j)
        else -> listOf(i, j * 2)
    }
    val (i, j) = newShape
    Backend.copyInto(
        x = value,
        y = result,
        yi = i,
        yj = j,
        axis = axis,
        indices = when (axis) {
            0 -> 0 until i step 2
            else -> 0 until j step 2
        },
    )
    Backend.copyInto(
        x = other.value,
        y = result,
        yi = i,
        yj = j,
        axis = axis,
        indices = when (axis) {
            0 -> 1 until i step 2
            else -> 1 until j step 2
        },
    )
    return IOType.D2(shape = newShape, value = result)
}
