package com.wsr.knist.batch.elementwise.operation.times

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchFloatTimesD0s")
operator fun Float.times(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.times(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchFloatTimesD1s")
operator fun Float.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.times(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchFloatTimesD2s")
operator fun Float.times(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.times(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchFloatTimesD3s")
operator fun Float.times(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.times(x = this, y = other.value)
    return Batch(size = other.size, shape = other.shape, value = result)
}

@JvmName("batchD0sTimesFloat")
operator fun Batch<IOType.D0>.times(other: Float): Batch<IOType.D0> {
    val result = Backend.times(x = value, y = other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sTimesD0s")
operator fun Batch<IOType.D0>.times(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.times(x = value, y = other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sTimesD1s")
operator fun Batch<IOType.D0>.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.times(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
    return Batch(size = size, shape = other.shape, value = result)
}

@JvmName("batchD0sTimesD2s")
operator fun Batch<IOType.D0>.times(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.times(
        x = value,
        y = other.value,
        yi = size,
        yj = other.step,
        axis = 0,
    )
    return Batch(size = size, shape = other.shape, value = result)
}

@JvmName("batchD0sTimesD3s")
operator fun Batch<IOType.D0>.times(other: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.times(
        x = value,
        y = other.value,
        yi = size,
        yj = other.step,
        axis = 0,
    )
    return Batch(size = size, shape = other.shape, value = result)
}
