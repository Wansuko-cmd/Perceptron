package com.wsr.batch.reshape.interleave

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.batch.Batch
import com.wsr.core.IOType

fun Batch<IOType.D1>.interleave(other: Batch<IOType.D1>): Batch<IOType.D1> {
    check(size == other.size && shape[0] == other.shape[0])
    val i = shape[0] * 2
    val result = DataBuffer.create(value.size + other.value.size)
    Backend.copyInto(
        x = value,
        y = result,
        yi = size,
        yj = i,
        axis = 1,
        indices = 0 until i step 2,
    )
    Backend.copyInto(
        x = other.value,
        y = result,
        yi = size,
        yj = i,
        axis = 1,
        indices = 1 until i step 2,
    )
    return Batch(size = size, shape = listOf(i), value = result)
}

fun Batch<IOType.D2>.interleave(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
    check(size == other.size && shape[0] == other.shape[0] && shape[1] == other.shape[1])
    val result = DataBuffer.create(value.size + other.value.size)
    val newShape = when (axis) {
        0 -> listOf(shape[0] * 2, shape[1])
        else -> listOf(shape[0], shape[1] * 2)
    }
    val (i, j) = newShape
    Backend.copyInto(
        x = value,
        y = result,
        yi = size,
        yj = i,
        yk = j,
        axis = axis + 1,
        indices = when (axis) {
            0 -> 0 until i step 2
            else -> 0 until j step 2
        },
    )
    Backend.copyInto(
        x = other.value,
        y = result,
        yi = size,
        yj = i,
        yk = j,
        axis = axis + 1,
        indices = when (axis) {
            0 -> 1 until i step 2
            else -> 1 until j step 2
        },
    )
    return Batch(
        size = size,
        shape = newShape,
        value = result,
    )
}
