package com.wsr.core.elementwise.math

import com.wsr.Backend
import com.wsr.core.IOType
import com.wsr.core.get
import kotlin.math.pow

fun IOType.D0.pow(n: Int): IOType.D0 {
    val result = Backend.pow(x = value, n = n)
    return IOType.D0(value = result)
}

fun IOType.D0.sqrt(e: Float = 1e-7f): IOType.D0 {
    val result = Backend.sqrt(x = value, e = e)
    return IOType.D0(value = result)
}
