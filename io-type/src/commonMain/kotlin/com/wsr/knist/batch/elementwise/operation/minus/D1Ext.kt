package com.wsr.knist.batch.elementwise.operation.minus

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d1
import com.wsr.knist.batch.d2
import com.wsr.knist.batch.get
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD1sMinusFloat")
operator fun Batch<IOType.D1>.minus(other: Float): Batch<IOType.D1> {
    val result = Backend.minus(x = value, y = other)
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sMinusD0s")
operator fun Batch<IOType.D1>.minus(other: Batch<IOType.D0>): Batch<IOType.D1> {
    val result = Backend.minus(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 0,
    )
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sMinusD1")
operator fun Batch<IOType.D1>.minus(other: IOType.D1): Batch<IOType.D1> {
    val result = Backend.minus(x = value, xi = size, xj = step, y = other.value, axis = 1)
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sMinusD1s")
operator fun Batch<IOType.D1>.minus(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.minus(x = value, y = other.value)
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sMinusD2sWithAxis")
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
    return Batch.d2(size, other.shape, result)
}
