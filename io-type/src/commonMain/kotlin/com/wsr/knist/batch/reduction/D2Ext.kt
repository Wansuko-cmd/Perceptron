package com.wsr.knist.batch.reduction

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD2sSum")
fun Batch<IOType.D2>.sum(): Batch<IOType.D0> {
    val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
    return Batch(size = size, shape = listOf(1), value = result)
}

@JvmName("batchD2sSumWithAxis")
fun Batch<IOType.D2>.sum(axis: Int): Batch<IOType.D1> {
    val result = Backend.sum(x = value, xi = size, xj = i, xk = j, axis = axis + 1)
    return Batch(size = size, shape = listOf(if (axis == 0) j else i), value = result)
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

@JvmName("batchD2sMaxIndexWithAxis")
fun Batch<IOType.D2>.maxIndex(axis: Int): Batch<IOType.D1> {
    val result = Backend.maxIndex(x = value, xi = size, xj = i, xk = j, axis = axis + 1)
    return Batch(size = size, shape = listOf(if (axis == 0) j else i), value = result)
}
