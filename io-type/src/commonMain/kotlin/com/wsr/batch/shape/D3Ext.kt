package com.wsr.batch.shape

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.get
import com.wsr.core.IOType
import kotlin.jvm.JvmName

fun Batch<IOType.D3>.toD4(): IOType.D4 = IOType.D4(shape = listOf(size, shape[0], shape[1], shape[2]), value = value)

fun IOType.D4.toBatch(): Batch<IOType.D3> =
    Batch(value = value, size = shape[0], shape = listOf(shape[1], shape[2], shape[3]))

@JvmName("batchD3sToList")
fun Batch<IOType.D3>.toList(): List<IOType.D3> = List(size) { get(it) }

@JvmName("batchD3sFlatten")
fun Batch<IOType.D3>.flatten() = Batch<IOType.D1>(
    shape = listOf(step),
    size = size,
    value = value,
)

@JvmName("batchD3sReshapeToD2")
fun Batch<IOType.D3>.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))

@JvmName("batchD3sReshapeToD2ByShape")
fun Batch<IOType.D3>.reshapeToD2(shape: List<Int>) = Batch<IOType.D2>(size = size, shape = shape, value = value)

fun Batch<IOType.D3>.transpose(axisI: Int, axisJ: Int, axisK: Int): Batch<IOType.D3> {
    val result = Backend.transpose(
        x = value,
        xi = size,
        xj = shape[0],
        xk = shape[1],
        xl = shape[2],
        axisI = 0,
        axisJ = axisI + 1,
        axisK = axisJ + 1,
        axisL = axisK + 1,
    )
    return Batch(size = size, shape = listOf(shape[axisI], shape[axisJ], shape[axisK]), value = result)
}
