package com.wsr.knist.batch.reduction.average

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.IOType
import com.wsr.knist.core.D3
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@JvmName("batchD3sAverage")
@ScopeOp
fun Batch<IOType.D3>.average(): Batch<IOType.D0> {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
    return Batch(size = size, shape = listOf(1), value = result)
}

@JvmName("batchD3sAverageWithAxis")
@ScopeOp
fun Batch<IOType.D3>.average(axis: Int): Batch<IOType.D2> = when (axis) {
    0 -> Batch(
        size = size,
        shape = listOf(j, k),
        value = Backend.average(x = value, xi = size, xj = i, xk = j * k, axis = 1),
    )

    1 -> Batch(
        size = size,
        shape = listOf(i, k),
        value = Backend.average(x = value, xi = size * i, xj = j, xk = k, axis = 1),
    )

    2 -> Batch(
        size = size,
        shape = listOf(i, j),
        value = Backend.average(x = value, xi = size, xj = i * j, xk = k, axis = 2),
    )

    else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
}

@JvmName("batchD3sBatchAverage")
@ScopeOp
fun Batch<IOType.D3>.batchAverage(): IOType.D3 {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
    return IOType.D3(shape = shape, value = result)
}
