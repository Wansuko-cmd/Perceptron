package com.wsr.knist.batch.linalg

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("batchD1sInnerD1s")
@ScopeOp
infix fun Batch<IOType.D1>.inner(other: Batch<IOType.D1>): Batch<IOType.D0> {
    val result = Backend.inner(x = value, y = other.value, b = size)
    return Batch(value = result, size = size, shape = listOf(1))
}
