package com.wsr.knist.batch.index

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d1
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType

fun IOType.D0.gather(other: Batch<IOType.D2>, axis: Int = 1): Batch<IOType.D1> = when (axis) {
    0 -> {
        val result = Backend.gather(x = value, y = other.value, i = other.size, j = other.i, k = other.j)
        Batch.d1(other.size, other.j, result)
    }

    else -> {
        val result = Backend.gather(
            x = value,
            y = other.value,
            i = other.size * other.i,
            j = other.j,
            k = 1,
        )
        Batch.d1(other.size, other.i, result)
    }
}
