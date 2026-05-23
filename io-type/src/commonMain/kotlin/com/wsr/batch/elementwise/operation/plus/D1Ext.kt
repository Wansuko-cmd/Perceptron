package com.wsr.batch.elementwise.operation.plus

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.i
import com.wsr.batch.j
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD1sPlusFloat")
operator fun Batch<IOType.D1>.plus(other: Float): Batch<IOType.D1> {
    val result = Backend.plus(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sPlusD0s")
operator fun Batch<IOType.D1>.plus(other: Batch<IOType.D0>): Batch<IOType.D1> {
    val result = Backend.plus(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 0,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sPlusD1")
operator fun Batch<IOType.D1>.plus(other: IOType.D1): Batch<IOType.D1> {
    val result = Backend.plus(x = value, xi = size, xj = step, y = other.value, axis = 1)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sPlusD1s")
operator fun Batch<IOType.D1>.plus(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.plus(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sPlusD2sWithAxis")
fun Batch<IOType.D1>.plus(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
    val result = Backend.plus(
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
