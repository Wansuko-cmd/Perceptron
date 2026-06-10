package com.wsr.knist.batch.shape

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

fun Batch<IOType.D3>.toD4(): IOType.D4 = IOType.D4(shape = listOf(size, i, j, k), value = value)

fun IOType.D4.toBatch(): Batch<IOType.D3> = Batch(value = value, size = i, shape = listOf(j, k, l))

@JvmName("batchD3sToList")
fun Batch<IOType.D3>.toList(): List<IOType.D3> = List(size) { get(it) }

@JvmName("batchD3sFlatten")
fun Batch<IOType.D3>.flatten() = Batch<IOType.D1>(
    shape = listOf(step),
    size = size,
    value = value,
)

@JvmName("batchD3sReshapeToD4")
fun Batch<IOType.D3>.reshapeToD4(i: Int, j: Int, k: Int, l: Int) =
    Batch<IOType.D4>(size = size, shape = listOf(i, j, k, l), value = value)

@JvmName("batchD3sReshapeToD2")
fun Batch<IOType.D3>.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))

@JvmName("batchD3sReshapeToD2ByShape")
fun Batch<IOType.D3>.reshapeToD2(shape: List<Int>) = Batch<IOType.D2>(size = size, shape = shape, value = value)

@ScopeOp
fun Batch<IOType.D3>.transpose(axisI: Int, axisJ: Int, axisK: Int): Batch<IOType.D3> {
    val result = Backend.transpose(
        x = value,
        xi = size,
        xj = i,
        xk = j,
        xl = k,
        axisI = 0,
        axisJ = axisI + 1,
        axisK = axisJ + 1,
        axisL = axisK + 1,
    )
    return Batch(size = size, shape = listOf(shape[axisI], shape[axisJ], shape[axisK]), value = result)
}
