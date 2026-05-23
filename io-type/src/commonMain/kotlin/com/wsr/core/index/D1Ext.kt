package com.wsr.core.index

import com.wsr.Backend
import com.wsr.core.IOType

fun IOType.D1.gather(other: IOType.D2): IOType.D2 {
    val result = Backend.gather(x = value, y = other.value, i = 1, j = other.i, k = other.j)
    return IOType.D2(shape = listOf(size, other.j), value = result)
}
