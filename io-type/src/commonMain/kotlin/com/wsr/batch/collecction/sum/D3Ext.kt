package com.wsr.batch.collecction.sum

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType

fun Batch<IOType.D3>.sum(): Batch<IOType.D0> {
    val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
    return Batch(size = size, shape = listOf(1), value = result)
}

fun Batch<IOType.D3>.sum(axis: Int): Batch<IOType.D2> = when (axis) {
    0 -> Batch(
        size = size,
        shape = listOf(shape[1], shape[2]),
        value = Backend.sum(x = value, xi = size, xj = shape[0], xk = shape[1] * shape[2], axis = 1),
    )
    1 -> Batch(
        size = size,
        shape = listOf(shape[0], shape[2]),
        value = Backend.sum(x = value, xi = size * shape[0], xj = shape[1], xk = shape[2], axis = 1),
    )
    2 -> Batch(
        size = size,
        shape = listOf(shape[0], shape[1]),
        value = Backend.sum(x = value, xi = size, xj = shape[0] * shape[1], xk = shape[2], axis = 2),
    )
    else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
}
