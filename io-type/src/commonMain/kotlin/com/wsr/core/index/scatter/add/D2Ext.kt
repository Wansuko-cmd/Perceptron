package com.wsr.core.index.scatter.add

import com.wsr.Backend
import com.wsr.core.IOType

fun IOType.D2.scatterAdd(other: IOType.D1, n: Int): IOType.D2 {
    val result = Backend.scatterAdd(x = value, y = other.value, i = 1, j = n, k = j, b = 1)
    return IOType.D2(shape = listOf(n, j), value = result)
}
