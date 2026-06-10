package com.wsr.knist.batch.elementwise.operation.div

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("batchFloatDivD0s")
operator fun Float.div(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.div(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchFloatDivD1s")
operator fun Float.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.div(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchFloatDivD2s")
operator fun Float.div(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.div(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchFloatDivD3s")
operator fun Float.div(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.div(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchD0sDivFloat")
@ScopeOp
operator fun Batch<IOType.D0>.div(other: Float): Batch<IOType.D0> {
    val result = Backend.div(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sDivD0s")
@ScopeOp
operator fun Batch<IOType.D0>.div(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.div(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sDivD1s")
@ScopeOp
operator fun Batch<IOType.D0>.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.div(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
    return Batch(size = size, shape = other.shape, value = result)
}

@JvmName("batchD0sDivD2s")
@ScopeOp
operator fun Batch<IOType.D0>.div(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.div(
        x = value,
        y = other.value,
        yi = size,
        yj = other.step,
        axis = 0,
    )
    return Batch(size = size, shape = other.shape, value = result)
}

@JvmName("batchD0sDivD3s")
@ScopeOp
operator fun Batch<IOType.D0>.div(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.div(
        x = value,
        y = other.value,
        yi = size,
        yj = other.step,
        axis = 0,
    )
    return Batch(size = size, shape = other.shape, value = result)
}
