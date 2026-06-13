package com.wsr.knist.batch.reduction

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d3
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.batch.l
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD4sMaxWithAxis")
fun Batch<IOType.D4>.max(axis: Int): Batch<IOType.D3> = when (axis) {
    0 -> Batch.d3(
        size,
        j,
        k,
        l,
        Backend.max(x = value, xi = size, xj = i, xk = j * k * l, axis = 1),
    )

    1 -> Batch.d3(
        size,
        i,
        k,
        l,
        Backend.max(x = value, xi = size * i, xj = j, xk = k * l, axis = 1),
    )

    2 -> Batch.d3(
        size,
        i,
        j,
        l,
        Backend.max(x = value, xi = size * i * j, xj = k, xk = l, axis = 1),
    )

    3 -> Batch.d3(
        size,
        i,
        j,
        k,
        Backend.max(x = value, xi = size, xj = i * j * k, xk = l, axis = 2),
    )

    else -> throw IllegalArgumentException("axis is $axis, not 0, 1, 2 or 3.")
}
