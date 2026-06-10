package com.wsr.knist.batch.elementwise.operation.minus

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("batchFloatMinusD0s")
operator fun Float.minus(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.minus(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchFloatMinusD1s")
operator fun Float.minus(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.minus(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchFloatMinusD2s")
operator fun Float.minus(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.minus(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchFloatMinusD3s")
operator fun Float.minus(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.minus(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchD0sMinusFloat")
@ScopeOp
operator fun Batch<IOType.D0>.minus(other: Float): Batch<IOType.D0> {
    val result = Backend.minus(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sMinusD0s")
@ScopeOp
operator fun Batch<IOType.D0>.minus(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.minus(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sMinusD1s")
@ScopeOp
operator fun Batch<IOType.D0>.minus(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.minus(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
    return Batch(size = size, shape = other.shape, value = result)
}

@JvmName("batchD0sMinusD2s")
@ScopeOp
operator fun Batch<IOType.D0>.minus(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.minus(
        x = value,
        y = other.value,
        yi = size,
        yj = other.step,
        axis = 0,
    )
    return Batch(size = size, shape = other.shape, value = result)
}

@JvmName("batchD0sMinusD3s")
@ScopeOp
operator fun Batch<IOType.D0>.minus(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.minus(
        x = value,
        y = other.value,
        yi = size,
        yj = other.step,
        axis = 0,
    )
    return Batch(size = size, shape = other.shape, value = result)
}
