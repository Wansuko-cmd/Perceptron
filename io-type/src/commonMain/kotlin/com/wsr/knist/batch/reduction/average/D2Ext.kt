package com.wsr.knist.batch.reduction.average

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d0
import com.wsr.knist.batch.d1
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD2sAverage")
fun Batch<IOType.D2>.average(): Batch<IOType.D0> {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
    return Batch.d0(size, result)
}

@JvmName("batchD2sAverageWithAxis")
fun Batch<IOType.D2>.average(axis: Int): Batch<IOType.D1> {
    val result = Backend.average(x = value, xi = size, xj = i, xk = j, axis = axis + 1)
    return Batch.d1(
        size,
        when (axis) {
            0 -> listOf(j)
            else -> listOf(i)
        },
        result,
    )
}

@JvmName("batchD2sBatchAverage")
fun Batch<IOType.D2>.batchAverage(): IOType.D2 {
    val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
    return IOType.D2(shape = shape, value = result)
}
