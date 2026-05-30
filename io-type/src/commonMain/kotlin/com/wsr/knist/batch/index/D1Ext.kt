package com.wsr.knist.batch.index

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.core.IOType

fun Batch<IOType.D1>.gather(other: IOType.D2): Batch<IOType.D2> {
    val result = Backend.gather(x = value, y = other.value, i = 1, j = other.i, k = other.j)
    return Batch(size = size, shape = listOf(i, other.j), value = result)
}
