package com.wsr.batch.reshape.slice

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.batch.Batch
import com.wsr.core.IOType
import com.wsr.create
import kotlin.math.min

fun Batch<IOType.D1>.slice(indices: IntProgression): Batch<IOType.D1> {
    val result = Backend.slice(x = value, xi = size, xj = shape[0], axis = 1, indices = indices)
    return Batch(size = size, shape = listOf(indices.size), value = result)
}

fun Batch<IOType.D2>.slice(indices: IntProgression, axis: Int): Batch<IOType.D2> {
    val result = Backend.slice(x = value, xi = size, xj = shape[0], xk = shape[1], axis = axis + 1, indices = indices)
    return Batch(
        size = size,
        shape = when (axis) {
            0 -> listOf(indices.size, shape[1])
            else -> listOf(shape[0], indices.size)
        },
        value = result,
    )
}
