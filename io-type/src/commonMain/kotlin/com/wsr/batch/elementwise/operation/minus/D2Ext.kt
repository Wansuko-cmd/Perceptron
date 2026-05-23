package com.wsr.batch.elementwise.operation.minus

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.i
import com.wsr.batch.j
import com.wsr.batch.k
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD2sMinusFloat")
operator fun Batch<IOType.D2>.minus(other: Float): Batch<IOType.D2> {
    val result = Backend.minus(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sMinusD0s")
operator fun Batch<IOType.D2>.minus(other: Batch<IOType.D0>): Batch<IOType.D2> {
    val result = Backend.minus(x = value, xi = size, xj = step, y = other.value, axis = 0)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sMinusD1WithAxis")
fun Batch<IOType.D2>.minus(other: IOType.D1, axis: Int): Batch<IOType.D2> {
    val result = Backend.minus(x = value, xi = size, xj = i, xk = j, y = other.value, axis = axis + 1)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sMinusD1sWithAxis")
fun Batch<IOType.D2>.minus(other: Batch<IOType.D1>, axis: Int): Batch<IOType.D2> {
    val result = Backend.minus(
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
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sMinusD2")
operator fun Batch<IOType.D2>.minus(other: IOType.D2): Batch<IOType.D2> {
    val result = Backend.minus(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 1,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sMinusD2s")
operator fun Batch<IOType.D2>.minus(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.minus(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sMinusD3sWithAxis")
fun Batch<IOType.D2>.minus(other: Batch<IOType.D3>, axis1: Int, axis2: Int): Batch<IOType.D3> {
    val result = Backend.minus(
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
    return Batch(size = other.size, shape = other.shape, value = result)
}
