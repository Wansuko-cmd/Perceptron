package com.wsr.batch.reduction.average

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.i
import com.wsr.batch.j
import com.wsr.batch.k
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
        shape = listOf(j, k),
        value = Backend.average(x = value, xi = size, xj = i, xk = j * k, axis = 1),
    )
    1 -> Batch(
        size = size,
        shape = listOf(i, k),
        value = Backend.average(x = value, xi = size * i, xj = j, xk = k, axis = 1),
    )
    2 -> Batch(
        size = size,
        shape = listOf(i, j),
        value = Backend.average(x = value, xi = size, xj = i * j, xk = k, axis = 2),
    )
    else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
}

@JvmName("batchD3sBatchAverage")
fun Batch<IOType.D3>.batchAverage(): IOType.D3 {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
    return IOType.D3(shape = shape, value = result)
}
