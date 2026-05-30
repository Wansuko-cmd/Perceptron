package com.wsr.knist.core.reduction

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType

fun IOType.D3.average(): IOType.D0 = IOType.D0(Backend.average(value))

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

fun IOType.D3.max(): IOType.D0 = IOType.D0(Backend.max(x = value))

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

fun IOType.D3.min() = IOType.D0(Backend.min(x = value))

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

fun IOType.D3.sum(): IOType.D0 = IOType.D0(Backend.sum(value))

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
