package com.wsr.knist.batch.linalg

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d1
import com.wsr.knist.batch.d2
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType
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

    return Batch.d1(other.size, n, result)
}

@JvmName("batchD2sMatMulD2")
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
    return Batch.d2(size, m, n, result)
}

@JvmName("batchD2sMatMulD2s")
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
    return Batch.d2(size, m, n, result)
}
