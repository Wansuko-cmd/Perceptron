package com.wsr.core.reshape.slice

import com.wsr.Backend
import com.wsr.base.data.size
import com.wsr.core.IOType

fun IOType.D1.slice(indices: IntProgression): IOType.D1 {
    val result = Backend.slice(x = value, indices = indices)
    return IOType.D1(value = result)
}

fun IOType.D2.slice(indices: IntProgression, axis: Int): IOType.D2 {
    val result = Backend.slice(x = value, xi = i, xj = j, axis = axis, indices = indices)
    return IOType.D2(
        shape = when (axis) {
            0 -> listOf(indices.size, j)
            else -> listOf(i, indices.size)
        },
        value = result,
    )
}
