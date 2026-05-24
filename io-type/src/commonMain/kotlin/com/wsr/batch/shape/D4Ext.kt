package com.wsr.batch.shape

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.i
import com.wsr.batch.j
import com.wsr.batch.k
import com.wsr.batch.l
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD4sReshapeToD3")
fun Batch<IOType.D4>.reshapeToD3(i: Int, j: Int, k: Int) = Batch<IOType.D3>(size = size, shape = listOf(i, j, k), value = value)

fun Batch<IOType.D4>.transpose(axisI: Int, axisJ: Int, axisK: Int, axisL: Int): Batch<IOType.D4> {
    val result = Backend.transpose(
        x = value,
        xi = size,
        xj = i,
        xk = j,
        xl = k,
        xm = l,
        axisI = 0,
        axisJ = axisI + 1,
        axisK = axisJ + 1,
        axisL = axisK + 1,
        axisM = axisL + 1,
    )
    return Batch(size = size, shape = listOf(shape[axisI], shape[axisJ], shape[axisK], shape[axisL]), value = result)
}
