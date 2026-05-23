package com.wsr.batch.linalg

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.i
import com.wsr.batch.j
import com.wsr.core.IOType
import kotlin.jvm.JvmName

fun IOType.D2.matMul(other: Batch<IOType.D1>, trans: Boolean = false): Batch<IOType.D1> {
    val n = if (trans) j else i
    val k = if (trans) i else j
    val result = Backend.matMul(
        x = other.value,
        transX = false,
        y = value,
        transY = !trans,
        m = other.size,
        n = n,
        k = k,
        b = 1,
    )

    return Batch(result, other.size, listOf(n))
}

@JvmName("matMulToD2s")
fun Batch<IOType.D2>.matMul(other: IOType.D2, transA: Boolean = false, transB: Boolean = false): Batch<IOType.D2> {
    val m = if (transA) j else i
    val n = if (transB) other.i else other.j
    val k = if (transA) i else j
    val result = Backend.matMul(
        x = value,
        transX = transA,
        y = other.value,
        transY = transB,
        m = size * m,
        n = n,
        k = k,
        b = 1,
    )
    return Batch(value = result, size = size, shape = listOf(m, n))
}

@JvmName("matMulToD2s")
fun Batch<IOType.D2>.matMul(
    other: Batch<IOType.D2>,
    transA: Boolean = false,
    transB: Boolean = false,
): Batch<IOType.D2> {
    val m = if (transA) j else i
    val n = if (transB) other.i else other.j
    val k = if (transA) i else j
    val result = Backend.matMul(
        x = value,
        transX = transA,
        y = other.value,
        transY = transB,
        m = m,
        n = n,
        k = k,
        b = size,
    )
    return Batch(value = result, size = size, shape = listOf(m, n))
}
