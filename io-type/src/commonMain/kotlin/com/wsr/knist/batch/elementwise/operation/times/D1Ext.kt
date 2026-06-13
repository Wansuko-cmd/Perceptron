package com.wsr.knist.batch.elementwise.operation.times

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d1
import com.wsr.knist.batch.d2
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@JvmName("batchD1TimesD1s")
operator fun IOType.D1.times(other: Batch<IOType.D1>): Batch<IOType.D1.Global> {
    val result = Backend.times(
        x = value,
        y = other.value,
        yi = other.size,
        yj = other.step,
        axis = 1,
    )
    return Batch.d1(other.size, other.shape, result)
}

@JvmName("batchD1sTimesFloat")
@ScopeOp
operator fun Batch<IOType.D1>.times(other: Float): Batch<IOType.D1.Global> {
    val result = Backend.times(x = value, y = other)
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sTimesD0s")
@ScopeOp
operator fun Batch<IOType.D1>.times(other: Batch<IOType.D0>): Batch<IOType.D1.Global> {
    val result = Backend.times(
        x = value,
        xi = size,
        xj = step,
        y = other.value,
        axis = 0,
    )
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sTimesD1")
@ScopeOp
operator fun Batch<IOType.D1>.times(other: IOType.D1): Batch<IOType.D1.Global> {
    val result = Backend.times(x = value, xi = size, xj = step, y = other.value, axis = 1)
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sTimesD1s")
@ScopeOp
operator fun Batch<IOType.D1>.times(other: Batch<IOType.D1>): Batch<IOType.D1.Global> {
    val result = Backend.times(x = value, y = other.value)
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sTimesD2sWithAxis")
@ScopeOp
fun Batch<IOType.D1>.times(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2.Global> {
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
    return Batch.d2(size, other.shape, result)
}
