package com.wsr.knist.batch.reduction

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD1sSum")
fun Batch<IOType.D1>.sum(): Batch<IOType.D0> {
    val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
    return Batch(shape = listOf(1), size = size, value = result)
}

@JvmName("batchD1sMax")
fun Batch<IOType.D1>.max(): Batch<IOType.D0> {
    val result = Backend.max(x = value, xi = size, xj = step, axis = 1)
    return Batch(shape = listOf(1), size = size, value = result)
}

@JvmName("batchD1sMin")
fun Batch<IOType.D1>.min(): Batch<IOType.D0> {
    val result = Backend.min(x = value, xi = size, xj = step, axis = 1)
    return Batch(shape = listOf(1), size = size, value = result)
}
