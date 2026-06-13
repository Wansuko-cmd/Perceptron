package com.wsr.knist.batch.shape

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.base.data.size
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d1
import com.wsr.knist.batch.d2
import com.wsr.knist.batch.d3
import com.wsr.knist.batch.d4
import com.wsr.knist.batch.get
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.D4
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

fun Batch<IOType.D3>.toD4(): IOType.D4 = IOType.D4(shape = listOf(size, i, j, k), value = value)

fun Batch<IOType.D3>.broadcastToD4(axis: Int, size: Int): Batch<IOType.D4> = when (axis) {
    0 -> {
        val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size, j = 1, k = i * j * k)
        Batch.d4(this.size, size, i, j, k, result)
    }

    1 -> {
        val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size * i, j = 1, k = j * k)
        Batch.d4(this.size, i, size, j, k, result)
    }

    2 -> {
        val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size * i * j, j = 1, k = k)
        Batch.d4(this.size, i, j, size, k, result)
    }

    3 -> {
        val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size * i * j * k, j = 1, k = 1)
        Batch.d4(this.size, i, j, k, size, result)
    }

    else -> throw IllegalArgumentException("Batch<IOType.D3>.broadcastToD4 axis is $axis not 0, 1, 2 or 3.")
}

fun IOType.D4.toBatch(): Batch<IOType.D3> = Batch.d3(i, j, k, l, value)

@JvmName("batchD3sToList")
fun Batch<IOType.D3>.toList(): List<IOType.D3> = List(size) { get(it) }

@JvmName("batchD3sFlatten")
fun Batch<IOType.D3>.flatten() = Batch.d1(size, step, value)

@JvmName("batchD3sReshapeToD4")
fun Batch<IOType.D3>.reshapeToD4(i: Int, j: Int, k: Int, l: Int) = Batch.d4(size, i, j, k, l, value)

@JvmName("batchD3sReshapeToD2")
fun Batch<IOType.D3>.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))

@JvmName("batchD3sReshapeToD2ByShape")
fun Batch<IOType.D3>.reshapeToD2(shape: List<Int>) = Batch.d2(size, shape, value)

fun Batch<IOType.D3>.slice(indices: IntProgression, axis: Int): Batch<IOType.D3> = when (axis) {
    0 -> {
        val result = Backend.slice(x = value, xi = size, xj = i, xk = j * k, axis = 1, indices = indices)
        Batch.d3(size, indices.size, j, k, result)
    }

    1 -> {
        val result = Backend.slice(x = value, xi = size * i, xj = j, xk = k, axis = 1, indices = indices)
        Batch.d3(size, i, indices.size, k, result)
    }

    2 -> {
        val result = Backend.slice(x = value, xi = size * i * j, xj = k, axis = 1, indices = indices)
        Batch.d3(size, i, j, indices.size, result)
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
    return Batch.d3(size, shape[axisI], shape[axisJ], shape[axisK], result)
}
