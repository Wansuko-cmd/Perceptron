package com.wsr.knist.batch.reduction

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d0
import com.wsr.knist.batch.d2
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
    return Batch.d0(size, result)
}

@JvmName("batchD3sSumWithAxis")
@ScopeOp
fun Batch<IOType.D3>.sum(axis: Int): Batch<IOType.D2> = when (axis) {
    0 -> Batch.d2(
        size,
        j,
        k,
        Backend.sum(x = value, xi = size, xj = i, xk = j * k, axis = 1),
    )

    1 -> Batch.d2(
        size,
        i,
        k,
        Backend.sum(x = value, xi = size * i, xj = j, xk = k, axis = 1),
    )

    2 -> Batch.d2(
        size,
        i,
        j,
        Backend.sum(x = value, xi = size, xj = i * j, xk = k, axis = 2),
    )

    else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
}

@JvmName("batchD3sMax")
@ScopeOp
fun Batch<IOType.D3>.max(): Batch<IOType.D0> {
    val result = Backend.max(x = value, xi = size, xj = step, axis = 1)
    return Batch.d0(size, result)
}

@JvmName("batchD3sMin")
@ScopeOp
fun Batch<IOType.D3>.min(): Batch<IOType.D0> {
    val result = Backend.min(x = value, xi = size, xj = step, axis = 1)
    return Batch.d0(size, result)
}

@JvmName("batchD3sMaxWithAxis")
fun Batch<IOType.D3>.max(axis: Int): Batch<IOType.D2> = when (axis) {
    0 -> Batch.d2(
        size,
        j,
        k,
        Backend.max(x = value, xi = size, xj = i, xk = j * k, axis = 1),
    )

    1 -> Batch.d2(
        size,
        i,
        k,
        Backend.max(x = value, xi = size * i, xj = j, xk = k, axis = 1),
    )

    2 -> Batch.d2(
        size,
        i,
        j,
        Backend.max(x = value, xi = size, xj = i * j, xk = k, axis = 2),
    )

    else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
}

@JvmName("batchD3sMaxIndexWithAxis")
@ScopeOp
fun Batch<IOType.D3>.maxIndex(axis: Int): Batch<IOType.D2> = when (axis) {
    0 -> Batch.d2(
        size,
        j,
        k,
        Backend.maxIndex(x = value, xi = size, xj = i, xk = j * k, axis = 1),
    )

    1 -> Batch.d2(
        size,
        i,
        k,
        Backend.maxIndex(x = value, xi = size * i, xj = j, xk = k, axis = 1),
    )

    2 -> Batch.d2(
        size,
        i,
        j,
        Backend.maxIndex(x = value, xi = size, xj = i * j, xk = k, axis = 2),
    )

    else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
}
