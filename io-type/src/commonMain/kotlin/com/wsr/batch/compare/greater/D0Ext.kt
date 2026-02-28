package com.wsr.batch.compare.greater

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD0sGtFloat")
infix fun Batch<IOType.D0>.gt(other: Float): Batch<IOType.D0> {
    val result = Backend.greaterThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sGtD0s")
infix fun Batch<IOType.D0>.gt(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.greaterThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}
