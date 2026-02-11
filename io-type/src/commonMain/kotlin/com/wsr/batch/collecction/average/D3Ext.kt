package com.wsr.batch.collecction.average

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD3sAverageBatch")
fun Batch<IOType.D3>.average(): Batch<IOType.D0> {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
    return Batch(size = size, shape = listOf(1), value = result)
}

@JvmName("batchD2sAverageWithAxis")
fun Batch<IOType.D3>.average(axis: Int): Batch<IOType.D2> = when (axis) {
    0 -> Batch(
        size = size,
        shape = listOf(shape[1], shape[2]),
        value = Backend.average(x = value, xi = size, xj = shape[0], xk = shape[1] * shape[2], axis = 1),
    )
    1 -> Batch(
        size = size,
        shape = listOf(shape[0], shape[2]),
        value = Backend.average(x = value, xi = size * shape[0], xj = shape[1], xk = shape[2], axis = 1),
    )
    2 -> Batch(
        size = size,
        shape = listOf(shape[0], shape[1]),
        value = Backend.average(x = value, xi = size, xj = shape[0] * shape[1], xk = shape[2], axis = 2),
    )
    else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
}

@JvmName("batchD3sBatchAverage")
fun Batch<IOType.D3>.batchAverage(): IOType.D3 {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
    return IOType.D3(shape = shape, value = result)
}
