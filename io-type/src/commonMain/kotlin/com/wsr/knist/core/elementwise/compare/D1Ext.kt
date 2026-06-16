package com.wsr.knist.core.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.core.D1
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import com.wsr.knist.scope.ScopeOpDefault

@ScopeOp
infix fun IOType.D1.eq(other: Float): IOType.D1.Global = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun IOType.D1.eq(
    other: Float,
    @ScopeOpDefault(EQUALS_ABSOLUTE_TOLERANCE_FQN) absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    @ScopeOpDefault(EQUALS_RELATIVE_TOLERANCE_FQN) relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D1.Global {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D1(result)
}

@ScopeOp
infix fun IOType.D1.eq(other: IOType.D1): IOType.D1.Global = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun IOType.D1.eq(
    other: IOType.D1,
    @ScopeOpDefault(EQUALS_ABSOLUTE_TOLERANCE_FQN) absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    @ScopeOpDefault(EQUALS_RELATIVE_TOLERANCE_FQN) relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D1.Global {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D1(result)
}

@ScopeOp
infix fun IOType.D1.gt(other: Float): IOType.D1.Global {
    val result = Backend.greaterThan(value, other)
    return IOType.D1(result)
}

@ScopeOp
infix fun IOType.D1.gt(other: IOType.D1): IOType.D1.Global {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D1(result)
}

@ScopeOp
infix fun IOType.D1.lt(other: Float): IOType.D1.Global {
    val result = Backend.lessThan(value, other)
    return IOType.D1(result)
}

@ScopeOp
infix fun IOType.D1.lt(other: IOType.D1): IOType.D1.Global {
    val result = Backend.lessThan(value, other.value)
    return IOType.D1(result)
}
