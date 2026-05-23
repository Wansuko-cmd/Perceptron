package com.wsr.core.shape

import com.wsr.Backend
import com.wsr.core.IOType

fun IOType.D3.transpose(axisI: Int, axisJ: Int, axisK: Int): IOType.D3 {
    val result = Backend.transpose(x = value, xi = i, xj = j, xk = k, axisI = axisI, axisJ = axisJ, axisK = axisK)
    return IOType.D3(shape = listOf(shape[axisI], shape[axisJ], shape[axisK]), value = result)
}

fun IOType.D3.reshapeToD2(i: Int, j: Int) = IOType.D2(shape = listOf(i, j), value = value)
