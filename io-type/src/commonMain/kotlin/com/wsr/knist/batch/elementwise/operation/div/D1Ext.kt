package com.wsr.knist.batch.elementwise.operation.div

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("batchD1sDivFloat")
@ScopeOp
operator fun Batch<IOType.D1>.div(other: Float): Batch<IOType.D1> {
    val result = Backend.div(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sDivD0s")
@ScopeOp
operator fun Batch<IOType.D1>.div(other: Batch<IOType.D0>): Batch<IOType.D1> {
    val result = Backend.div(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 0,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sDivD1")
@ScopeOp
operator fun Batch<IOType.D1>.div(other: IOType.D1): Batch<IOType.D1> {
    val result = Backend.div(x = value, xi = size, xj = step, y = other.value, axis = 1)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sDivD1s")
@ScopeOp
operator fun Batch<IOType.D1>.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.div(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sDivD2sWithAxis")
@ScopeOp
fun Batch<IOType.D1>.div(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
    val result = Backend.div(
        x = value,
        xi = size,
        xj = i,
        y = other.value,
        yi = other.size,
        yj = other.i,
        yk = other.j,
        axis1 = 0,
        axis2 = axis + 1,
    )
    return Batch(size = size, shape = other.shape, value = result)
}
