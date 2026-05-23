package com.wsr.core.shape

import com.wsr.Backend
import com.wsr.core.IOType
import com.wsr.core.d4
import com.wsr.core.get

fun IOType.D4.transpose(axisI: Int, axisJ: Int, axisK: Int, axisL: Int): IOType.D4 {
    val result = Backend.transpose(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        xl = l,
        axisI = axisI,
        axisJ = axisJ,
        axisK = axisK,
        axisL = axisL,
    )
    return IOType.D4(shape = listOf(shape[axisI], shape[axisJ], shape[axisK], shape[axisL]), value = result)
}

fun IOType.D4.reshapeToD2(i: Int, j: Int) = IOType.D2(shape = listOf(i, j), value = value)

fun IOType.D4.reshapeToD3(i: Int, j: Int, k: Int) = IOType.D3(shape = listOf(i, j, k), value = value)
