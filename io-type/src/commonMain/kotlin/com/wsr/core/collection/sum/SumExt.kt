package com.wsr.core.collection.sum

import com.wsr.Backend
import com.wsr.core.IOType

fun IOType.D1.sum(): IOType.D0 = IOType.D0(Backend.sum(x = value))

fun IOType.D2.sum(): IOType.D0 = IOType.D0(Backend.sum(x = value))

fun IOType.D2.sum(axis: Int): IOType.D1 {
    val result = Backend.sum(x = value, xi = i, xj = j, axis = axis)
    return IOType.D1(result)
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

fun IOType.D4.sum(): IOType.D0 = IOType.D0(Backend.sum(value))
