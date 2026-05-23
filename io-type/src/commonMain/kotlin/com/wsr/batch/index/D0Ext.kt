package com.wsr.batch.index

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.i
import com.wsr.batch.j
import com.wsr.core.IOType

fun IOType.D0.gather(other: Batch<IOType.D2>, axis: Int = 1): Batch<IOType.D1> = when (axis) {
    0 -> {
        val result = Backend.gather(x = value, y = other.value, i = other.size, j = other.i, k = other.j)
        Batch(
            size = other.size,
            shape = listOf(other.j),
            value = result,
        )
    }
    else -> {
        val result = Backend.gather(
            x = value,
            y = other.value,
            i = other.size * other.i,
            j = other.j,
            k = 1,
        )
        Batch(
            size = other.size,
            shape = listOf(other.i),
            value = result,
        )
    }
}
