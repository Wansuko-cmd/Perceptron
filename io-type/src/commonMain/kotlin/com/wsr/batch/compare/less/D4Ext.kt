package com.wsr.batch.compare.less

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD4sLtFloat")
infix fun Batch<IOType.D4>.lt(other: Float): Batch<IOType.D4> {
    val result = Backend.lessThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD4sLtD4s")
infix fun Batch<IOType.D4>.lt(other: Batch<IOType.D4>): Batch<IOType.D4> {
    val result = Backend.lessThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}
