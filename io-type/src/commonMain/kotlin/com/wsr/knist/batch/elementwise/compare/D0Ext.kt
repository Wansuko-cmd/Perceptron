package com.wsr.knist.batch.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.compare.EQUALS_ABSOLUTE_TOLERANCE
import com.wsr.knist.core.elementwise.compare.EQUALS_RELATIVE_TOLERANCE
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("infixBatchD0sEqFloat")
@ScopeOp
infix fun Batch<IOType.D0>.eq(other: Float): Batch<IOType.D0> = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun Batch<IOType.D0>.eq(
    other: Float,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): Batch<IOType.D0> {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("infixBatchD0sEqD0s")
@ScopeOp
infix fun Batch<IOType.D0>.eq(other: Batch<IOType.D0>): Batch<IOType.D0> = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@JvmName("batchD0sEqD0s")
@ScopeOp
fun Batch<IOType.D0>.eq(
    other: Batch<IOType.D0>,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): Batch<IOType.D0> {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sGtFloat")
@ScopeOp
infix fun Batch<IOType.D0>.gt(other: Float): Batch<IOType.D0> {
    val result = Backend.greaterThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sGtD0s")
@ScopeOp
infix fun Batch<IOType.D0>.gt(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.greaterThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sLtFloat")
@ScopeOp
infix fun Batch<IOType.D0>.lt(other: Float): Batch<IOType.D0> {
    val result = Backend.lessThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sLtD0s")
@ScopeOp
infix fun Batch<IOType.D0>.lt(other: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.lessThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}
