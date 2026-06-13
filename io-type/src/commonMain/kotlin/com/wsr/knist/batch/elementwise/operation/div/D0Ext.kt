package com.wsr.knist.batch.elementwise.operation.div

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d0
import com.wsr.knist.batch.d1
import com.wsr.knist.batch.d2
import com.wsr.knist.batch.d3
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@JvmName("batchFloatDivD0s")
operator fun Float.div(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.div(x = this, y = other.value)
    return Batch.d0(other.size, result)
}

@JvmName("batchFloatDivD1s")
operator fun Float.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.div(x = this, y = other.value)
    return Batch.d1(other.size, other.shape, result)
}

@JvmName("batchFloatDivD2s")
operator fun Float.div(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.div(x = this, y = other.value)
    return Batch.d2(other.size, other.shape, result)
}

@JvmName("batchFloatDivD3s")
operator fun Float.div(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.div(x = this, y = other.value)
    return Batch.d3(other.size, other.shape, result)
}

@JvmName("batchD0sDivFloat")
@ScopeOp
operator fun Batch<IOType.D0>.div(other: Float): Batch<IOType.D0> {
    val result = Backend.div(x = value, y = other)
    return Batch.d0(size, result)
}

@JvmName("batchD0sDivD0s")
@ScopeOp
operator fun Batch<IOType.D0>.div(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.div(x = value, y = other.value)
    return Batch.d0(size, result)
}

@JvmName("batchD0sDivD1s")
@ScopeOp
operator fun Batch<IOType.D0>.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.div(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
    return Batch.d1(size, other.shape, result)
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
    return Batch.d2(size, other.shape, result)
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
    return Batch.d3(size, other.shape, result)
}
