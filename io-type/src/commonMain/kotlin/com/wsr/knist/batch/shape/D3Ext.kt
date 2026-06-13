package com.wsr.knist.batch.shape

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.base.data.size
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

fun Batch<IOType.D3>.toD4(): IOType.D4 = IOType.D4(shape = listOf(size, i, j, k), value = value)

fun Batch<IOType.D3>.broadcastToD4(axis: Int, size: Int): Batch<IOType.D4> = when (axis) {
    0 -> {
        val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size, j = 1, k = i * j * k)
        Batch(size = this.size, shape = listOf(size, i, j, k), value = result)
    }

    1 -> {
        val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size * i, j = 1, k = j * k)
        Batch(size = this.size, shape = listOf(i, size, j, k), value = result)
    }

    2 -> {
        val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size * i * j, j = 1, k = k)
        Batch(size = this.size, shape = listOf(i, j, size, k), value = result)
    }

    3 -> {
        val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size * i * j * k, j = 1, k = 1)
        Batch(size = this.size, shape = listOf(i, j, k, size), value = result)
    }

    else -> throw IllegalArgumentException("Batch<IOType.D3>.broadcastToD4 axis is $axis not 0, 1, 2 or 3.")
}

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

fun Batch<IOType.D3>.slice(indices: IntProgression, axis: Int): Batch<IOType.D3> = when (axis) {
    0 -> {
        val result = Backend.slice(x = value, xi = size, xj = i, xk = j * k, axis = 1, indices = indices)
        Batch(size = size, shape = listOf(indices.size, j, k), value = result)
    }

    1 -> {
        val result = Backend.slice(x = value, xi = size * i, xj = j, xk = k, axis = 1, indices = indices)
        Batch(size = size, shape = listOf(i, indices.size, k), value = result)
    }

    2 -> {
        val result = Backend.slice(x = value, xi = size * i * j, xj = k, axis = 1, indices = indices)
        Batch(size = size, shape = listOf(i, j, indices.size), value = result)
    }

    else -> throw IllegalArgumentException("Batch<IOType.D3>.slice axis is $axis, not 0, 1 or 2.")
}

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
