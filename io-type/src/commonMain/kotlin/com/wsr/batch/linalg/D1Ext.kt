package com.wsr.batch.linalg

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD1sInnerD1s")
infix fun Batch<IOType.D1>.inner(other: Batch<IOType.D1>): Batch<IOType.D0> {
    val result = Backend.inner(x = value, y = other.value, b = size)
    return Batch(value = result, size = size, shape = listOf(1))
}
