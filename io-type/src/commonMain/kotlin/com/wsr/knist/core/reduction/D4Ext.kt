package com.wsr.knist.core.reduction

import com.wsr.knist.Backend
import com.wsr.knist.core.D0
import com.wsr.knist.core.D3
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp

@ScopeOp
fun IOType.D4.average(): IOType.D0 = IOType.D0(Backend.average(value))

@ScopeOp
fun IOType.D4.average(axis: Int): IOType.D3 = when (axis) {
    0 -> IOType.D3(
        shape = listOf(j, k, l),
        value = Backend.average(x = value, xi = i, xj = j, xk = k * l, axis = 0),
    )

    1 -> IOType.D3(
        shape = listOf(i, k, l),
        value = Backend.average(x = value, xi = i, xj = j, xk = k * l, axis = 1),
    )

    2 -> IOType.D3(
        shape = listOf(i, j, l),
        value = Backend.average(x = value, xi = i * j, xj = k, xk = l, axis = 1),
    )

    else -> IOType.D3(
        shape = listOf(i, j, k),
        value = Backend.average(x = value, xi = i * j, xj = k, xk = l, axis = 2),
    )
}

@ScopeOp
fun IOType.D4.max(): IOType.D0 = IOType.D0(Backend.max(x = value))

@ScopeOp
fun IOType.D4.max(axis: Int): IOType.D3 = when (axis) {
    0 -> IOType.D3(
        shape = listOf(j, k, l),
        value = Backend.max(x = value, xi = i, xj = j, xk = k * l, axis = 0),
    )

    1 -> IOType.D3(
        shape = listOf(i, k, l),
        value = Backend.max(x = value, xi = i, xj = j, xk = k * l, axis = 1),
    )

    2 -> IOType.D3(
        shape = listOf(i, j, l),
        value = Backend.max(x = value, xi = i * j, xj = k, xk = l, axis = 1),
    )

    else -> IOType.D3(
        shape = listOf(i, j, k),
        value = Backend.max(x = value, xi = i * j, xj = k, xk = l, axis = 2),
    )
}

@ScopeOp
fun IOType.D4.min() = IOType.D0(Backend.min(x = value))

@ScopeOp
fun IOType.D4.min(axis: Int): IOType.D3 = when (axis) {
    0 -> IOType.D3(
        shape = listOf(j, k, l),
        value = Backend.min(x = value, xi = i, xj = j, xk = k * l, axis = 0),
    )

    1 -> IOType.D3(
        shape = listOf(i, k, l),
        value = Backend.min(x = value, xi = i, xj = j, xk = k * l, axis = 1),
    )

    2 -> IOType.D3(
        shape = listOf(i, j, l),
        value = Backend.min(x = value, xi = i * j, xj = k, xk = l, axis = 1),
    )

    else -> IOType.D3(
        shape = listOf(i, j, k),
        value = Backend.min(x = value, xi = i * j, xj = k, xk = l, axis = 2),
    )
}

@ScopeOp
fun IOType.D4.sum(): IOType.D0 = IOType.D0(Backend.sum(value))

@ScopeOp
fun IOType.D4.sum(axis: Int): IOType.D3 = when (axis) {
    0 -> IOType.D3(
        shape = listOf(j, k, l),
        value = Backend.sum(x = value, xi = i, xj = j, xk = k * l, axis = 0),
    )

    1 -> IOType.D3(
        shape = listOf(i, k, l),
        value = Backend.sum(x = value, xi = i, xj = j, xk = k * l, axis = 1),
    )

    2 -> IOType.D3(
        shape = listOf(i, j, l),
        value = Backend.sum(x = value, xi = i * j, xj = k, xk = l, axis = 1),
    )

    else -> IOType.D3(
        shape = listOf(i, j, k),
        value = Backend.sum(x = value, xi = i * j, xj = k, xk = l, axis = 2),
    )
}

@ScopeOp
fun IOType.D4.maxIndex(axis: Int): IOType.D3 = when (axis) {
    0 -> IOType.D3(
        shape = listOf(j, k, l),
        value = Backend.maxIndex(x = value, xi = i, xj = j, xk = k * l, axis = 0),
    )

    1 -> IOType.D3(
        shape = listOf(i, k, l),
        value = Backend.maxIndex(x = value, xi = i, xj = j, xk = k * l, axis = 1),
    )

    2 -> IOType.D3(
        shape = listOf(i, j, l),
        value = Backend.maxIndex(x = value, xi = i * j, xj = k, xk = l, axis = 1),
    )

    else -> IOType.D3(
        shape = listOf(i, j, k),
        value = Backend.maxIndex(x = value, xi = i * j, xj = k, xk = l, axis = 2),
    )
}
