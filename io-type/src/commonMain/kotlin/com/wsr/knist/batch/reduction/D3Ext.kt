package com.wsr.knist.batch.reduction

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@JvmName("batchD3sSum")
@ScopeOp
fun Batch<IOType.D3>.sum(): Batch<IOType.D0> {
    val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
    return Batch(size = size, shape = listOf(1), value = result)
}

@JvmName("batchD3sSumWithAxis")
@ScopeOp
fun Batch<IOType.D3>.sum(axis: Int): Batch<IOType.D2> = when (axis) {
    0 -> Batch(
        size = size,
        shape = listOf(j, k),
        value = Backend.sum(x = value, xi = size, xj = i, xk = j * k, axis = 1),
    )
    1 -> Batch(
        size = size,
        shape = listOf(i, k),
        value = Backend.sum(x = value, xi = size * i, xj = j, xk = k, axis = 1),
    )
    2 -> Batch(
        size = size,
        shape = listOf(i, j),
        value = Backend.sum(x = value, xi = size, xj = i * j, xk = k, axis = 2),
    )
    else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
}

@JvmName("batchD3sMax")
@ScopeOp
fun Batch<IOType.D3>.max(): Batch<IOType.D0> {
    val result = Backend.max(x = value, xi = size, xj = step, axis = 1)
    return Batch(shape = listOf(1), size = size, value = result)
}

@JvmName("batchD3sMin")
@ScopeOp
fun Batch<IOType.D3>.min(): Batch<IOType.D0> {
    val result = Backend.min(x = value, xi = size, xj = step, axis = 1)
    return Batch(shape = listOf(1), size = size, value = result)
}

@JvmName("batchD3sMaxIndexWithAxis")
@ScopeOp
fun Batch<IOType.D3>.maxIndex(axis: Int): Batch<IOType.D2> = when (axis) {
    0 -> Batch(
        size = size,
        shape = listOf(j, k),
        value = Backend.maxIndex(x = value, xi = size, xj = i, xk = j * k, axis = 1),
    )
    1 -> Batch(
        size = size,
        shape = listOf(i, k),
        value = Backend.maxIndex(x = value, xi = size * i, xj = j, xk = k, axis = 1),
    )
    2 -> Batch(
        size = size,
        shape = listOf(i, j),
        value = Backend.maxIndex(x = value, xi = size, xj = i * j, xk = k, axis = 2),
    )
    else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
}
