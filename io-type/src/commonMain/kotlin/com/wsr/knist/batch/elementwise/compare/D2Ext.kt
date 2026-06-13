package com.wsr.knist.batch.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d2
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.compare.EQUALS_ABSOLUTE_TOLERANCE
import com.wsr.knist.core.elementwise.compare.EQUALS_RELATIVE_TOLERANCE
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@JvmName("infixBatchD2sEqFloat")
@ScopeOp
infix fun Batch<IOType.D2>.eq(other: Float): Batch<IOType.D2> = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun Batch<IOType.D2>.eq(
    other: Float,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): Batch<IOType.D2> {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return Batch.d2(size, shape, result)
}

@JvmName("infixBatchD2sEqD2s")
@ScopeOp
infix fun Batch<IOType.D2>.eq(other: Batch<IOType.D2>): Batch<IOType.D2> = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@JvmName("batchD2sEqD2s")
@ScopeOp
fun Batch<IOType.D2>.eq(
    other: Batch<IOType.D2>,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): Batch<IOType.D2> {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sGtFloat")
@ScopeOp
infix fun Batch<IOType.D2>.gt(other: Float): Batch<IOType.D2> {
    val result = Backend.greaterThan(value, other)
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sGtD2s")
@ScopeOp
infix fun Batch<IOType.D2>.gt(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.greaterThan(value, other.value)
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sLtFloat")
@ScopeOp
infix fun Batch<IOType.D2>.lt(other: Float): Batch<IOType.D2> {
    val result = Backend.lessThan(value, other)
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sLtD2s")
@ScopeOp
infix fun Batch<IOType.D2>.lt(other: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.lessThan(value, other.value)
    return Batch.d2(size, shape, result)
}
