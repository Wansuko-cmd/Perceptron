package com.wsr.knist.batch.reduction.average

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("batchD4sBatchAverage")
@ScopeOp
fun Batch<IOType.D4>.batchAverage(): IOType.D4 {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
    return IOType.D4(shape = shape, value = result)
}
