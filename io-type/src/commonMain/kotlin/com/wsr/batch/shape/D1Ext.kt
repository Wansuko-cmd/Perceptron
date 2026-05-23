package com.wsr.batch.shape

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.batch.Batch
import com.wsr.batch.get
import com.wsr.batch.i
import com.wsr.core.IOType
import com.wsr.core.shape.broadcastToD2
import kotlin.jvm.JvmName

fun Batch<IOType.D1>.broadcastToD2(axis: Int, size: Int) = Batch(this.size) { this[it].broadcastToD2(axis, size) }

fun Batch<IOType.D1>.toD2(): IOType.D2 = IOType.D2(shape = listOf(size, i), value = value)

fun IOType.D2.toBatch(): Batch<IOType.D1> = Batch(size = i, shape = listOf(j), value = value)

@JvmName("batchD1sReshapeToD2")
fun Batch<IOType.D1>.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))

@JvmName("batchD1sReshapeToD2ByShape")
fun Batch<IOType.D1>.reshapeToD2(shape: List<Int>) = Batch<IOType.D2>(size = size, shape = shape, value = value)

@JvmName("batchD1sReshapeToD3")
fun Batch<IOType.D1>.reshapeToD3(i: Int, j: Int, k: Int) = reshapeToD3(listOf(i, j, k))

@JvmName("batchD1sReshapeToD3ByShape")
fun Batch<IOType.D1>.reshapeToD3(shape: List<Int>) = Batch<IOType.D3>(size = size, shape = shape, value = value)

fun Batch<IOType.D1>.slice(indices: IntProgression): Batch<IOType.D1> {
    val result = Backend.slice(x = value, xi = size, xj = i, axis = 1, indices = indices)
    return Batch(size = size, shape = listOf(indices.size), value = result)
}

fun Batch<IOType.D1>.interleave(other: Batch<IOType.D1>): Batch<IOType.D1> {
    check(size == other.size && i == other.shape[0])
    val i = shape[0] * 2
    val result = DataBuffer.create(value.size + other.value.size)
    Backend.copyInto(
        x = value,
        y = result,
        yi = size,
        yj = i,
        axis = 1,
        indices = 0 until i step 2,
    )
    Backend.copyInto(
        x = other.value,
        y = result,
        yi = size,
        yj = i,
        axis = 1,
        indices = 1 until i step 2,
    )
    return Batch(size = size, shape = listOf(i), value = result)
}
