package com.wsr.knist.batch.reduction.average

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("batchD2sAverage")
@ScopeOp
fun Batch<IOType.D2>.average(): Batch<IOType.D0> {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
    return Batch(size = size, shape = listOf(1), value = result)
}

@JvmName("batchD2sAverageWithAxis")
@ScopeOp
fun Batch<IOType.D2>.average(axis: Int): Batch<IOType.D1> {
    val result = Backend.average(x = value, xi = size, xj = i, xk = j, axis = axis + 1)
    return Batch(
        size = size,
        shape = when (axis) {
            0 -> listOf(j)
            else -> listOf(i)
        },
        value = result,
    )
}

@JvmName("batchD2sBatchAverage")
@ScopeOp
fun Batch<IOType.D2>.batchAverage(): IOType.D2 {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
    return IOType.D2(shape = shape, value = result)
}
