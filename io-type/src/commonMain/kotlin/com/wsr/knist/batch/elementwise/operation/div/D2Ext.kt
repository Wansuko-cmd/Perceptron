package com.wsr.knist.batch.elementwise.operation.div

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d2
import com.wsr.knist.batch.d3
import com.wsr.knist.batch.get
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD2sDivFloat")
operator fun Batch<IOType.D2>.div(other: Float): Batch<IOType.D2> {
    val result = Backend.div(x = value, y = other)
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sDivD0s")
operator fun Batch<IOType.D2>.div(other: Batch<IOType.D0>): Batch<IOType.D2> {
    val result = Backend.div(x = value, xi = size, xj = step, y = other.value, axis = 0)
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sDivD1WithAxis")
fun Batch<IOType.D2>.div(other: IOType.D1, axis: Int): Batch<IOType.D2> {
    val result = Backend.div(x = value, xi = size, xj = i, xk = j, y = other.value, axis = axis + 1)
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sDivD1sWithAxis")
fun Batch<IOType.D2>.div(other: Batch<IOType.D1>, axis: Int): Batch<IOType.D2> {
    val result = Backend.div(
        x = value,
        xi = size,
        xj = i,
        xk = j,
        y = other.value,
        yi = other.size,
        yj = other.i,
        axis1 = 0,
        axis2 = axis + 1,
    )
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sDivD2")
operator fun Batch<IOType.D2>.div(other: IOType.D2): Batch<IOType.D2> {
    val result = Backend.div(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 1,
    )
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sDivD2s")
operator fun Batch<IOType.D2>.div(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.div(x = value, y = other.value)
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sDivD3sWithAxis")
fun Batch<IOType.D2>.div(other: Batch<IOType.D3>, axis1: Int, axis2: Int): Batch<IOType.D3> {
    val result = Backend.div(
        x = value,
        xi = size,
        xj = i,
        xk = j,
        y = other.value,
        yi = other.size,
        yj = other.i,
        yk = other.j,
        yl = other.k,
        axis1 = 0,
        axis2 = axis1 + 1,
        axis3 = axis2 + 1,
    )
    return Batch.d3(other.size, other.shape, result)
}
