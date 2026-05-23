package com.wsr.batch.shape

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.batch.Batch
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.shape.broadcastToD3
import kotlin.jvm.JvmName

fun Batch<IOType.D2>.broadcastToD3(axis: Int, size: Int) = Batch(this.size) { this[it].broadcastToD3(axis, size) }

fun Batch<IOType.D2>.toD3(): IOType.D3 = IOType.D3(shape = listOf(size, shape[0], shape[1]), value = value)

fun IOType.D3.toBatch(): Batch<IOType.D2> = Batch(value = value, size = shape[0], shape = listOf(shape[1], shape[2]))

@JvmName("batchD2sToList")
fun Batch<IOType.D2>.toList(): List<IOType.D2> = List(size) { get(it) }

@JvmName("batchD2sFlatten")
fun Batch<IOType.D2>.flatten() = Batch<IOType.D1>(
    shape = listOf(step),
    size = size,
    value = value,
)

@JvmName("batchD2sReshapeToD3")
fun Batch<IOType.D2>.reshapeToD3(i: Int, j: Int, k: Int) = reshapeToD3(listOf(i, j, k))

@JvmName("batchD2sReshapeToD3ByShape")
fun Batch<IOType.D2>.reshapeToD3(shape: List<Int>) = Batch<IOType.D3>(size = size, shape = shape, value = value)

fun Batch<IOType.D2>.slice(indices: IntProgression, axis: Int): Batch<IOType.D2> {
    val result = Backend.slice(x = value, xi = size, xj = shape[0], xk = shape[1], axis = axis + 1, indices = indices)
    return Batch(
        size = size,
        shape = when (axis) {
            0 -> listOf(indices.size, shape[1])
            else -> listOf(shape[0], indices.size)
        },
        value = result,
    )
}

fun Batch<IOType.D2>.interleave(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
    check(size == other.size && shape[0] == other.shape[0] && shape[1] == other.shape[1])
    val result = DataBuffer.create(value.size + other.value.size)
    val newShape = when (axis) {
        0 -> listOf(shape[0] * 2, shape[1])
        else -> listOf(shape[0], shape[1] * 2)
    }
    val (i, j) = newShape
    Backend.copyInto(
        x = value,
        y = result,
        yi = size,
        yj = i,
        yk = j,
        axis = axis + 1,
        indices = when (axis) {
            0 -> 0 until i step 2
            else -> 0 until j step 2
        },
    )
    Backend.copyInto(
        x = other.value,
        y = result,
        yi = size,
        yj = i,
        yk = j,
        axis = axis + 1,
        indices = when (axis) {
            0 -> 1 until i step 2
            else -> 1 until j step 2
        },
    )
    return Batch(size = size, shape = newShape, value = result)
}

fun Batch<IOType.D2>.transpose(): Batch<IOType.D2> {
    val result = Backend.transpose(x = value, xi = size, xj = shape[0], xk = shape[1], axisI = 0, axisJ = 2, axisK = 1)
    return Batch(size = size, shape = shape.reversed(), value = result)
}
