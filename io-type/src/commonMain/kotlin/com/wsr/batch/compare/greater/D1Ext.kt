package com.wsr.batch.compare.greater

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD1sGtFloat")
infix fun Batch<IOType.D1>.gt(other: Float): Batch<IOType.D1> {
    val result = Backend.greaterThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sGtD1s")
infix fun Batch<IOType.D1>.gt(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.greaterThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}
