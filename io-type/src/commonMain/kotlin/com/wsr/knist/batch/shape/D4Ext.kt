package com.wsr.knist.batch.shape

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d3
import com.wsr.knist.batch.d4
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.batch.l
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@JvmName("batchD4sReshapeToD3")
fun Batch<IOType.D4>.reshapeToD3(i: Int, j: Int, k: Int) = Batch.d3(size, i, j, k, value)

@ScopeOp
fun Batch<IOType.D4>.transpose(axisI: Int, axisJ: Int, axisK: Int, axisL: Int): Batch<IOType.D4.Global> {
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
    return Batch.d4(size, shape[axisI], shape[axisJ], shape[axisK], shape[axisL], result)
}
