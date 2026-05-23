package com.wsr.batch.reduction

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD2sSum")
fun Batch<IOType.D2>.sum(): Batch<IOType.D0> {
    val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
    return Batch(size = size, shape = listOf(1), value = result)
}

@JvmName("batchD2sSumWithAxis")
fun Batch<IOType.D2>.sum(axis: Int): Batch<IOType.D1> {
    val result = Backend.sum(x = value, xi = size, xj = shape[0], xk = shape[1], axis = axis + 1)
    return Batch(size = size, shape = listOf(if (axis == 0) shape[1] else shape[0]), value = result)
}

@JvmName("batchD2sMax")
fun Batch<IOType.D2>.max(): Batch<IOType.D0> {
    val result = Backend.max(x = value, xi = size, xj = step, axis = 1)
    return Batch(shape = listOf(1), size = size, value = result)
}

@JvmName("batchD2sMin")
fun Batch<IOType.D2>.min(): Batch<IOType.D0> {
    val result = Backend.min(x = value, xi = size, xj = step, axis = 1)
    return Batch(shape = listOf(1), size = size, value = result)
}
