package com.wsr.knist.core.reduction

import com.wsr.knist.Backend
import com.wsr.knist.core.D0
import com.wsr.knist.core.D2
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp

@ScopeOp
fun IOType.D3.average(): IOType.D0 = IOType.D0(Backend.average(value))

@ScopeOp
fun IOType.D3.average(axis: Int): IOType.D2 {
    val result = Backend.average(x = value, xi = i, xj = j, xk = k, axis = axis)
    return IOType.D2(
        shape = when (axis) {
            0 -> listOf(j, k)
            1 -> listOf(i, k)
            else -> listOf(i, j)
        },
        value = result,
    )
}

@ScopeOp
fun IOType.D3.max(): IOType.D0 = IOType.D0(Backend.max(x = value))

@ScopeOp
fun IOType.D3.max(axis: Int): IOType.D2 {
    val result = Backend.max(x = value, xi = i, xj = j, xk = k, axis = axis)
    return IOType.D2(
        shape = when (axis) {
            0 -> listOf(j, k)
            1 -> listOf(i, k)
            else -> listOf(i, j)
        },
        value = result,
    )
}

@ScopeOp
fun IOType.D3.min() = IOType.D0(Backend.min(x = value))

@ScopeOp
fun IOType.D3.min(axis: Int): IOType.D2 {
    val result = Backend.min(x = value, xi = i, xj = j, xk = k, axis = axis)
    return IOType.D2(
        shape = when (axis) {
            0 -> listOf(j, k)
            1 -> listOf(i, k)
            else -> listOf(i, j)
        },
        value = result,
    )
}

@ScopeOp
fun IOType.D3.sum(): IOType.D0 = IOType.D0(Backend.sum(value))

@ScopeOp
fun IOType.D3.sum(axis: Int): IOType.D2 {
    val result = Backend.sum(x = value, xi = i, xj = j, xk = k, axis = axis)
    return IOType.D2(
        shape = when (axis) {
            0 -> listOf(j, k)
            1 -> listOf(i, k)
            else -> listOf(i, j)
        },
        value = result,
    )
}

@ScopeOp
fun IOType.D3.maxIndex(axis: Int): IOType.D2 {
    val result = Backend.maxIndex(x = value, xi = i, xj = j, xk = k, axis = axis)
    return IOType.D2(
        shape = when (axis) {
            0 -> listOf(j, k)
            1 -> listOf(i, k)
            else -> listOf(i, j)
        },
        value = result,
    )
}
