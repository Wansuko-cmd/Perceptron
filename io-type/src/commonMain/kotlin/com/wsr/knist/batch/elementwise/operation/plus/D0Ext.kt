package com.wsr.knist.batch.elementwise.operation.plus

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d0
import com.wsr.knist.batch.d1
import com.wsr.knist.batch.d2
import com.wsr.knist.batch.d3
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchFloatPlusD0s")
operator fun Float.plus(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.plus(x = this, y = other.value)
    return Batch.d0(other.size, result)
}

@JvmName("batchFloatPlusD1s")
operator fun Float.plus(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.plus(x = this, y = other.value)
    return Batch.d1(other.size, other.shape, result)
}

@JvmName("batchFloatPlusD2s")
operator fun Float.plus(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.plus(x = this, y = other.value)
    return Batch.d2(other.size, other.shape, result)
}

@JvmName("batchFloatPlusD3s")
operator fun Float.plus(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.plus(x = this, y = other.value)
    return Batch.d3(other.size, other.shape, result)
}

@JvmName("batchD0sPlusFloat")
operator fun Batch<IOType.D0>.plus(other: Float): Batch<IOType.D0> {
    val result = Backend.plus(x = value, y = other)
    return Batch.d0(size, result)
}

@JvmName("batchD0sPlusD0s")
operator fun Batch<IOType.D0>.plus(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.plus(x = value, y = other.value)
    return Batch.d0(size, result)
}

@JvmName("batchD0sPlusD1s")
operator fun Batch<IOType.D0>.plus(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.plus(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
    return Batch.d1(size, other.shape, result)
}

@JvmName("batchD0sPlusD2s")
operator fun Batch<IOType.D0>.plus(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.plus(
        x = value,
        y = other.value,
        yi = size,
        yj = other.step,
        axis = 0,
    )
    return Batch.d2(size, other.shape, result)
}

@JvmName("batchD0sPlusD3s")
operator fun Batch<IOType.D0>.plus(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.plus(
        x = value,
        y = other.value,
        yi = size,
        yj = other.step,
        axis = 0,
    )
    return Batch.d3(size, other.shape, result)
}
