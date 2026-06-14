package com.wsr.knist.core.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.core.D0
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import com.wsr.knist.scope.ScopeOpDefault

@PublishedApi
internal const val EQUALS_ABSOLUTE_TOLERANCE = 1e-4f

@PublishedApi
internal const val EQUALS_RELATIVE_TOLERANCE = 1e-4f

internal const val EQUALS_ABSOLUTE_TOLERANCE_FQN =
    "com.wsr.knist.core.elementwise.compare.EQUALS_ABSOLUTE_TOLERANCE"
internal const val EQUALS_RELATIVE_TOLERANCE_FQN =
    "com.wsr.knist.core.elementwise.compare.EQUALS_RELATIVE_TOLERANCE"

@ScopeOp
infix fun IOType.D0.eq(other: Float): IOType.D0.Global = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun IOType.D0.eq(
    other: Float,
    @ScopeOpDefault(EQUALS_ABSOLUTE_TOLERANCE_FQN) absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    @ScopeOpDefault(EQUALS_RELATIVE_TOLERANCE_FQN) relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D0.Global {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D0(result)
}

@ScopeOp
infix fun IOType.D0.eq(other: IOType.D0): IOType.D0.Global = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun IOType.D0.eq(
    other: IOType.D0,
    @ScopeOpDefault(EQUALS_ABSOLUTE_TOLERANCE_FQN) absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    @ScopeOpDefault(EQUALS_RELATIVE_TOLERANCE_FQN) relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D0.Global {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D0(result)
}

@ScopeOp
infix fun IOType.D0.gt(other: Float): IOType.D0.Global {
    val result = Backend.greaterThan(value, other)
    return IOType.D0(result)
}

@ScopeOp
infix fun IOType.D0.gt(other: IOType.D0): IOType.D0.Global {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D0(result)
}

@ScopeOp
infix fun IOType.D0.lt(other: Float): IOType.D0.Global {
    val result = Backend.lessThan(value, other)
    return IOType.D0(result)
}

@ScopeOp
infix fun IOType.D0.lt(other: IOType.D0): IOType.D0.Global {
    val result = Backend.lessThan(value, other.value)
    return IOType.D0(result)
}
