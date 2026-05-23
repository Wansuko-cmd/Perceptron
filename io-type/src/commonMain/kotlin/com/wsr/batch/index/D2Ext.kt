package com.wsr.batch.index

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType

fun Batch<IOType.D2>.scatterAdd(other: Batch<IOType.D1>, n: Int): IOType.D2 {
    val result = Backend.scatterAdd(x = value, y = other.value, i = 1, j = n, k = shape[1], b = size)
    return IOType.D2(shape = listOf(n, shape[1]), value = result)
}
