package com.wsr.knist.batch.elementwise.operation.minus

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@JvmName("batchD1sMinusFloat")
@ScopeOp
operator fun Batch<IOType.D1>.minus(other: Float): Batch<IOType.D1> {
    val result = Backend.minus(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sMinusD0s")
@ScopeOp
operator fun Batch<IOType.D1>.minus(other: Batch<IOType.D0>): Batch<IOType.D1> {
    val result = Backend.minus(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 0,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sMinusD1")
@ScopeOp
operator fun Batch<IOType.D1>.minus(other: IOType.D1): Batch<IOType.D1> {
    val result = Backend.minus(x = value, xi = size, xj = step, y = other.value, axis = 1)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sMinusD1s")
@ScopeOp
operator fun Batch<IOType.D1>.minus(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.minus(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sMinusD2sWithAxis")
@ScopeOp
fun Batch<IOType.D1>.minus(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
    val result = Backend.minus(
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
