package com.wsr.knist.batch.elementwise.operation.div

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("batchD3sDivFloat")
@ScopeOp
operator fun Batch<IOType.D3>.div(other: Float): Batch<IOType.D3> {
    val result = Backend.div(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sDivD0s")
@ScopeOp
operator fun Batch<IOType.D3>.div(other: Batch<IOType.D0>): Batch<IOType.D3> {
    val result = Backend.div(x = value, xi = size, xj = step, y = other.value, axis = 0)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sDivD1WithAxis")
@ScopeOp
fun Batch<IOType.D3>.div(other: IOType.D1, axis: Int): Batch<IOType.D3> {
    val result = Backend.div(
        x = value,
        xi = size,
        xj = i,
        xk = j,
        xl = k,
        y = other.value,
        axis = axis + 1,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sDivD2")
@ScopeOp
operator fun Batch<IOType.D3>.div(other: IOType.D2): Batch<IOType.D3> = div(other = other, axis1 = 1, axis2 = 2)

@JvmName("batchD3sDivD2WithAxis")
@ScopeOp
fun Batch<IOType.D3>.div(other: IOType.D2, axis1: Int, axis2: Int): Batch<IOType.D3> {
    val result = Backend.div(
        x = value,
        xi = size,
        xj = i,
        xk = j,
        xl = k,
        y = other.value,
        yi = other.i,
        yj = other.j,
        axis1 = axis1 + 1,
        axis2 = axis2 + 1,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sDivD2s")
@ScopeOp
operator fun Batch<IOType.D3>.div(other: Batch<IOType.D2>) = div(other, axis1 = 1, axis2 = 2)

@JvmName("batchD3sDivD2sWithAxis")
@ScopeOp
fun Batch<IOType.D3>.div(other: Batch<IOType.D2>, axis1: Int, axis2: Int): Batch<IOType.D3> {
    val result = Backend.div(
        x = value,
        xi = size,
        xj = i,
        xk = j,
        xl = k,
        y = other.value,
        yi = other.size,
        yj = other.i,
        yk = other.j,
        axis1 = 0,
        axis2 = axis1 + 1,
        axis3 = axis2 + 1,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sDivD3")
@ScopeOp
operator fun Batch<IOType.D3>.div(other: IOType.D3): Batch<IOType.D3> {
    val result = Backend.div(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 1,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sDivD3s")
@ScopeOp
operator fun Batch<IOType.D3>.div(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.div(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}
