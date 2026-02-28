package com.wsr.batch.compare.greater

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD3sGtFloat")
infix fun Batch<IOType.D3>.gt(other: Float): Batch<IOType.D3> {
    val result = Backend.greaterThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sGtD3s")
infix fun Batch<IOType.D3>.gt(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.greaterThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}
