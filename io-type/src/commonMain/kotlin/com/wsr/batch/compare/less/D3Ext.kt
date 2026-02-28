package com.wsr.batch.compare.less

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD3sLtFloat")
infix fun Batch<IOType.D3>.lt(other: Float): Batch<IOType.D3> {
    val result = Backend.lessThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sLtD3s")
infix fun Batch<IOType.D3>.lt(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.lessThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}
