package com.wsr.knist.batch.elementwise.operation.times

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD1TimesD1s")
operator fun IOType.D1.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.times(
        x = value,
        y = other.value,
        yi = other.size,
        yj = other.step,
        axis = 1,
    )
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchD1sTimesFloat")
operator fun Batch<IOType.D1>.times(other: Float): Batch<IOType.D1> {
    val result = Backend.times(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sTimesD0s")
operator fun Batch<IOType.D1>.times(other: Batch<IOType.D0>): Batch<IOType.D1> {
    val result = Backend.times(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 0,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sTimesD1")
operator fun Batch<IOType.D1>.times(other: IOType.D1): Batch<IOType.D1> {
    val result = Backend.times(x = value, xi = size, xj = step, y = other.value, axis = 1)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sTimesD1s")
operator fun Batch<IOType.D1>.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.times(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sTimesD2sWithAxis")
fun Batch<IOType.D1>.times(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
    val result = Backend.times(
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
