package com.wsr.batch.compare.greater

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD4sGtFloat")
infix fun Batch<IOType.D4>.gt(other: Float): Batch<IOType.D4> {
    val result = Backend.greaterThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD4sGtD4s")
infix fun Batch<IOType.D4>.gt(other: Batch<IOType.D4>): Batch<IOType.D4> {
    val result = Backend.greaterThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}
