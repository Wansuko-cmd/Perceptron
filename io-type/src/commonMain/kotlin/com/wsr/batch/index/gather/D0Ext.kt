package com.wsr.batch.index.gather

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType

fun IOType.D0.gather(other: Batch<IOType.D2>, axis: Int = 1): Batch<IOType.D1> = when (axis) {
    0 -> {
        val result = Backend.gather(x = value, y = other.value, i = other.size, j = other.shape[0], k = other.shape[1])
        Batch(
            size = other.size,
            shape = listOf(other.shape[1]),
            value = result,
        )
    }
    else -> {
        val result = Backend.gather(
            x = value,
            y = other.value,
            i = other.size * other.shape[0],
            j = other.shape[1],
            k = 1,
        )
        Batch(
            size = other.size,
            shape = listOf(other.shape[0]),
            value = result,
        )
    }
}
