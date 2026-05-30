package com.wsr.knist.batch.index

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType

fun Batch<IOType.D2>.scatterAdd(other: Batch<IOType.D1>, n: Int): IOType.D2 {
    val result = Backend.scatterAdd(x = value, y = other.value, i = 1, j = n, k = j, b = size)
    return IOType.D2(shape = listOf(n, j), value = result)
}
