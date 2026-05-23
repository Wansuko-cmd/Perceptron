package com.wsr.core.shape

import com.wsr.core.IOType
import com.wsr.core.d4
import com.wsr.core.get

fun IOType.D4.transpose(axisI: Int, axisJ: Int, axisK: Int, axisL: Int): IOType.D4 {
    val axes = listOf(axisI, axisJ, axisK, axisL)
    return IOType.d4(i = shape[axisI], j = shape[axisJ], k = shape[axisK], l = shape[axisL]) { i, j, k, l ->
        val indices = listOf(i, j, k, l)
        this[
            indices[axes.indexOf(0)],
            indices[axes.indexOf(1)],
            indices[axes.indexOf(2)],
            indices[axes.indexOf(3)],
        ]
    }
}

fun IOType.D4.reshapeToD2(i: Int, j: Int) = IOType.D2(shape = listOf(i, j), value = value)

fun IOType.D4.reshapeToD3(i: Int, j: Int, k: Int) = IOType.D3(shape = listOf(i, j, k), value = value)
