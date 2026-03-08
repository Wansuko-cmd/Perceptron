package com.wsr.batch.index.gather

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType

fun Batch<IOType.D1>.gather(other: IOType.D2): Batch<IOType.D2> {
    val result = Backend.gather(x = value, y = other.value, i = 1, j = other.i, k = other.j)
    return Batch(size = size, shape = listOf(shape[0], other.j), value = result)
}
