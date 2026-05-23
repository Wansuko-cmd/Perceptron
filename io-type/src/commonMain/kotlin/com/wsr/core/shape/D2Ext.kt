package com.wsr.core.shape

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.core.IOType
import com.wsr.core.d3
import com.wsr.core.get

fun IOType.D2.broadcastToD3(axis: Int, size: Int) = when (axis) {
    0 -> IOType.d3(size, shape[0], shape[1]) { i, j, k -> this[j, k] }
    1 -> IOType.d3(shape[0], size, shape[1]) { i, j, k -> this[i, k] }
    2 -> IOType.d3(shape[0], shape[1], size) { i, j, k -> this[i, j] }
    else -> throw IllegalArgumentException("IOType.D2.broadcastToD3 axis is $axis not 0, 1 or 2.")
}

fun IOType.D2.transpose(): IOType.D2 {
    val result = Backend.transpose(x = value, xi = i, xj = j)
    return IOType.D2(shape = listOf(j, i), value = result)
}

fun IOType.D2.reshapeToD3(i: Int, j: Int, k: Int) = reshapeToD3(shape = listOf(i, j, k))

fun IOType.D2.reshapeToD3(shape: List<Int>) = IOType.D3(shape = shape, value = value)

fun IOType.D2.reshapeToD4(i: Int, j: Int, k: Int, l: Int) = IOType.D4(shape = listOf(i, j, k, l), value = value)

fun IOType.D2.slice(indices: IntProgression, axis: Int): IOType.D2 {
    val result = Backend.slice(x = value, xi = i, xj = j, axis = axis, indices = indices)
    return IOType.D2(
        shape = when (axis) {
            0 -> listOf(indices.size, j)
            else -> listOf(i, indices.size)
        },
        value = result,
    )
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
