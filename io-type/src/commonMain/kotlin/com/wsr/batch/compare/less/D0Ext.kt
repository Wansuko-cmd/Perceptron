package com.wsr.batch.compare.less

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD0sLtFloat")
infix fun Batch<IOType.D0>.lt(other: Float): Batch<IOType.D0> {
    val result = Backend.lessThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sLtD0s")
infix fun Batch<IOType.D0>.lt(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.lessThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}
