package com.wsr.core.elementwise.math

import com.wsr.Backend
import com.wsr.core.IOType
import com.wsr.core.get
import kotlin.math.pow

fun IOType.D4.pow(n: Int): IOType.D4 {
    val result = Backend.pow(x = value, n = n)
    return IOType.D4(shape = shape, value = result)
}

fun IOType.D4.sqrt(e: Float = 1e-7f): IOType.D4 {
    val result = Backend.sqrt(x = value, e = e)
    return IOType.D4(shape = shape, value = result)
}
