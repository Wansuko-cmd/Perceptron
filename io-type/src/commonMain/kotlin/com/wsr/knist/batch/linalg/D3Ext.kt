package com.wsr.knist.batch.linalg

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

fun IOType.D3.matMul(other: Batch<IOType.D3>, transA: Boolean = false, transB: Boolean = false): Batch<IOType.D3> {
    val m = if (transA) k else j
    val n = if (transB) other.j else other.k
    val k = if (transA) j else k
    val result = Backend.matMul(
        x = value,
        transX = transA,
        y = other.value,
        transY = transB,
        m = m,
        n = size * n,
        k = k,
        b = i,
    )
    return Batch(value = result, size = size, shape = listOf(i, m, n))
}

@JvmName("batchD3sMatMulD3")
fun Batch<IOType.D3>.matMul(other: IOType.D3, transA: Boolean = false, transB: Boolean = false): Batch<IOType.D3> {
    val m = if (transA) k else j
    val n = if (transB) other.j else other.k
    val k = if (transA) j else k
    val result = Backend.matMul(
        x = value,
        transX = transA,
        y = other.value,
        transY = transB,
        m = size * m,
        n = n,
        k = k,
        b = i,
    )
    return Batch(value = result, size = size, shape = listOf(i, m, n))
}

@JvmName("batchD3sMatMulD3s")
fun Batch<IOType.D3>.matMul(
    other: Batch<IOType.D3>,
    transA: Boolean = false,
    transB: Boolean = false,
): Batch<IOType.D3> {
    val m = if (transA) this.k else this.j
    val n = if (transB) other.j else other.k
    val k = if (transA) this.j else this.k
    val result = Backend.matMul(
        x = value,
        transX = transA,
        y = other.value,
        transY = transB,
        m = m,
        n = n,
        k = k,
        b = size * i,
    )
    return Batch(value = result, size = size, shape = listOf(i, m, n))
}
