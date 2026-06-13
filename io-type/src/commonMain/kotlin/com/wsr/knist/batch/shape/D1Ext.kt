package com.wsr.knist.batch.shape

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.base.data.size
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.batch.i
import com.wsr.knist.core.D2
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@ScopeOp
fun Batch<IOType.D1>.broadcastToD2(axis: Int, size: Int): Batch<IOType.D2> = when (axis) {
    0 -> {
        val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size, j = 1, k = i)
        Batch(size = this.size, shape = listOf(size, i), value = result)
    }

    1 -> {
        val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size * i, j = 1, k = 1)
        Batch(size = this.size, shape = listOf(i, size), value = result)
    }

    else -> throw IllegalArgumentException("Batch<IOType.D1>.broadcastToD2 axis is $axis not 0 or 1.")
}

fun Batch<IOType.D1>.toD2(): IOType.D2 = IOType.D2(shape = listOf(size, i), value = value)

fun IOType.D2.toBatch(): Batch<IOType.D1> = Batch(size = i, shape = listOf(j), value = value)

@JvmName("batchD1sToList")
fun Batch<IOType.D1>.toList(): List<IOType.D1> = List(size) { get(it) }

@JvmName("batchD1sReshapeToD2")
fun Batch<IOType.D1>.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))

@JvmName("batchD1sReshapeToD2ByShape")
fun Batch<IOType.D1>.reshapeToD2(shape: List<Int>) = Batch<IOType.D2>(size = size, shape = shape, value = value)

@JvmName("batchD1sReshapeToD3")
fun Batch<IOType.D1>.reshapeToD3(i: Int, j: Int, k: Int) = reshapeToD3(listOf(i, j, k))

@JvmName("batchD1sReshapeToD3ByShape")
fun Batch<IOType.D1>.reshapeToD3(shape: List<Int>) = Batch<IOType.D3>(size = size, shape = shape, value = value)

@ScopeOp
fun Batch<IOType.D1>.slice(indices: IntProgression): Batch<IOType.D1> {
    val result = Backend.slice(x = value, xi = size, xj = i, axis = 1, indices = indices)
    return Batch(size = size, shape = listOf(indices.size), value = result)
}

@ScopeOp
fun Batch<IOType.D1>.interleave(other: Batch<IOType.D1>): Batch<IOType.D1> {
    check(size == other.size && i == other.i)
    val newI = i * 2
    val result = DataBuffer.create(value.size + other.value.size)
    Backend.copyInto(
        x = value,
        y = result,
        yi = size,
        yj = newI,
        axis = 1,
        indices = 0 until newI step 2,
    )
    Backend.copyInto(
        x = other.value,
        y = result,
        yi = size,
        yj = newI,
        axis = 1,
        indices = 1 until newI step 2,
    )
    return Batch(size = size, shape = listOf(newI), value = result)
}
