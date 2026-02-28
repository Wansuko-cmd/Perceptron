package com.wsr.batch.compare.greater

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD2sGtFloat")
infix fun Batch<IOType.D2>.gt(other: Float): Batch<IOType.D2> {
    val result = Backend.greaterThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sGtD2s")
infix fun Batch<IOType.D2>.gt(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.greaterThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}
