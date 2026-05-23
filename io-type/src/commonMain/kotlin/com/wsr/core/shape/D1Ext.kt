package com.wsr.core.shape

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.core.IOType
import com.wsr.core.d2
import com.wsr.core.get

fun IOType.D1.broadcastToD2(axis: Int, size: Int) = when (axis) {
    0 -> IOType.d2(size, shape[0]) { x, y -> this[y] }
    1 -> IOType.d2(shape[0], size) { x, y -> this[x] }
    else -> throw IllegalArgumentException("IOType.D1.broadcastToD2 axis is $axis not 0 or 1.")
}

fun IOType.D1.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))

fun IOType.D1.reshapeToD2(shape: List<Int>) = IOType.D2(shape = shape, value = value)

fun IOType.D1.slice(indices: IntProgression): IOType.D1 {
    val result = Backend.slice(x = value, indices = indices)
    return IOType.D1(value = result)
}

fun IOType.D1.interleave(other: IOType.D1): IOType.D1 {
    check(size == other.size)

    val result = DataBuffer.create(size + other.size)
    Backend.copyInto(x = value, y = result, indices = 0 until size * 2 step 2)
    Backend.copyInto(x = other.value, y = result, indices = 1 until size * 2 step 2)

    return IOType.D1(result)
}
