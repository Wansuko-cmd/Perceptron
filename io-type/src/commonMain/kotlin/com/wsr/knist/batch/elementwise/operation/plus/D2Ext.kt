package com.wsr.knist.batch.elementwise.operation.plus

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@JvmName("batchD2sPlusFloat")
@ScopeOp
operator fun Batch<IOType.D2>.plus(other: Float): Batch<IOType.D2> {
    val result = Backend.plus(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sPlusD0s")
@ScopeOp
operator fun Batch<IOType.D2>.plus(other: Batch<IOType.D0>): Batch<IOType.D2> {
    val result = Backend.plus(x = value, xi = size, xj = step, y = other.value, axis = 0)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sPlusD1WithAxis")
@ScopeOp
fun Batch<IOType.D2>.plus(other: IOType.D1, axis: Int): Batch<IOType.D2> {
    val result = Backend.plus(x = value, xi = size, xj = i, xk = j, y = other.value, axis = axis + 1)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sPlusD1sWithAxis")
@ScopeOp
fun Batch<IOType.D2>.plus(other: Batch<IOType.D1>, axis: Int): Batch<IOType.D2> {
    val result = Backend.plus(
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

@JvmName("batchD2sPlusD2")
@ScopeOp
operator fun Batch<IOType.D2>.plus(other: IOType.D2): Batch<IOType.D2> {
    val result = Backend.plus(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 1,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sPlusD2s")
@ScopeOp
operator fun Batch<IOType.D2>.plus(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.plus(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sPlusD3sWithAxis")
@ScopeOp
fun Batch<IOType.D2>.plus(other: Batch<IOType.D3>, axis1: Int, axis2: Int): Batch<IOType.D3> {
    val result = Backend.plus(
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
