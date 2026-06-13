package com.wsr.knist.batch.elementwise.operation.times

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD2TimesD2s")
operator fun IOType.D2.times(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.times(
        x = value,
        y = other.value,
        yi = other.size,
        yj = other.step,
        axis = 1,
    )
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchD2sTimesFloat")
operator fun Batch<IOType.D2>.times(other: Float): Batch<IOType.D2> {
    val result = Backend.times(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sTimesD0s")
operator fun Batch<IOType.D2>.times(other: Batch<IOType.D0>): Batch<IOType.D2> {
    val result = Backend.times(x = value, xi = size, xj = step, y = other.value, axis = 0)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sTimesD1WithAxis")
fun Batch<IOType.D2>.times(other: IOType.D1, axis: Int): Batch<IOType.D2> {
    val result = Backend.times(x = value, xi = size, xj = i, xk = j, y = other.value, axis = axis + 1)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sTimesD1sWithAxis")
fun Batch<IOType.D2>.times(other: Batch<IOType.D1>, axis: Int): Batch<IOType.D2> {
    val result = Backend.times(
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

@JvmName("batchD2sTimesD2")
operator fun Batch<IOType.D2>.times(other: IOType.D2): Batch<IOType.D2> {
    val result = Backend.times(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 1,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sTimesD2s")
operator fun Batch<IOType.D2>.times(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.times(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sTimesD3sWithAxis")
fun Batch<IOType.D2>.times(other: Batch<IOType.D3>, axis1: Int, axis2: Int): Batch<IOType.D3> {
    val result = Backend.times(
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
