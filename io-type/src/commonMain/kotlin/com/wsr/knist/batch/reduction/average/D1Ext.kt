package com.wsr.knist.batch.reduction.average

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.D1
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@JvmName("batchD1sAverage")
@ScopeOp
fun Batch<IOType.D1>.average(): Batch<IOType.D0> {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
    return Batch(size = size, shape = listOf(1), value = result)
}

@JvmName("batchD1sBatchAverage")
@ScopeOp
fun Batch<IOType.D1>.batchAverage(): IOType.D1 {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
    return IOType.D1(value = result)
}
