package com.wsr.knist.batch.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d1
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.compare.EQUALS_ABSOLUTE_TOLERANCE
import com.wsr.knist.core.elementwise.compare.EQUALS_ABSOLUTE_TOLERANCE_FQN
import com.wsr.knist.core.elementwise.compare.EQUALS_RELATIVE_TOLERANCE
import com.wsr.knist.core.elementwise.compare.EQUALS_RELATIVE_TOLERANCE_FQN
import com.wsr.knist.scope.ScopeOp
import com.wsr.knist.scope.ScopeOpDefault
import kotlin.jvm.JvmName

@JvmName("infixBatchD1sEqFloat")
@ScopeOp
infix fun Batch<IOType.D1>.eq(other: Float): Batch<IOType.D1.Global> = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun Batch<IOType.D1>.eq(
    other: Float,
    @ScopeOpDefault(EQUALS_ABSOLUTE_TOLERANCE_FQN)absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    @ScopeOpDefault(EQUALS_RELATIVE_TOLERANCE_FQN)relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): Batch<IOType.D1.Global> {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return Batch.d1(size, shape, result)
}

@JvmName("infixBatchD1sEqD1s")
@ScopeOp
infix fun Batch<IOType.D1>.eq(other: Batch<IOType.D1>): Batch<IOType.D1.Global> = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@JvmName("batchD1sEqD1s")
@ScopeOp
fun Batch<IOType.D1>.eq(
    other: Batch<IOType.D1>,
    @ScopeOpDefault(EQUALS_ABSOLUTE_TOLERANCE_FQN)absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    @ScopeOpDefault(EQUALS_RELATIVE_TOLERANCE_FQN)relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): Batch<IOType.D1.Global> {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sGtFloat")
@ScopeOp
infix fun Batch<IOType.D1>.gt(other: Float): Batch<IOType.D1.Global> {
    val result = Backend.greaterThan(value, other)
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sGtD1s")
@ScopeOp
infix fun Batch<IOType.D1>.gt(other: Batch<IOType.D1>): Batch<IOType.D1.Global> {
    val result = Backend.greaterThan(value, other.value)
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sLtFloat")
@ScopeOp
infix fun Batch<IOType.D1>.lt(other: Float): Batch<IOType.D1.Global> {
    val result = Backend.lessThan(value, other)
    return Batch.d1(size, shape, result)
}

@JvmName("batchD1sLtD1s")
@ScopeOp
infix fun Batch<IOType.D1>.lt(other: Batch<IOType.D1>): Batch<IOType.D1.Global> {
    val result = Backend.lessThan(value, other.value)
    return Batch.d1(size, shape, result)
}
