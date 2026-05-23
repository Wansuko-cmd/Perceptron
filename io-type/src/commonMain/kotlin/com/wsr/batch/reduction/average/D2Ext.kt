package com.wsr.batch.reduction.average

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.i
import com.wsr.batch.j
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD2sAverageBatch")
fun Batch<IOType.D2>.average(): Batch<IOType.D0> {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
    return Batch(size = size, shape = listOf(1), value = result)
}

@JvmName("batchD2sAverageWithAxis")
fun Batch<IOType.D2>.average(axis: Int): Batch<IOType.D1> {
    val result = Backend.average(x = value, xi = size, xj = i, xk = j, axis = axis + 1)
    return Batch(
        size = size,
        shape = when (axis) {
            0 -> listOf(j)
            else -> listOf(i)
        },
        value = result,
    )
}

@JvmName("batchD2sBatchAverage")
fun Batch<IOType.D2>.batchAverage(): IOType.D2 {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
    return IOType.D2(shape = shape, value = result)
}
