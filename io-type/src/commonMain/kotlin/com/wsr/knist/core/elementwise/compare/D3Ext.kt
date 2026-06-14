package com.wsr.knist.core.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.core.D2
import com.wsr.knist.core.D3
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import com.wsr.knist.scope.ScopeOpDefault

@ScopeOp
infix fun IOType.D3.eq(other: Float): IOType.D3.Global = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun IOType.D3.eq(
    other: Float,
    @ScopeOpDefault(EQUALS_ABSOLUTE_TOLERANCE_FQN) absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    @ScopeOpDefault(EQUALS_RELATIVE_TOLERANCE_FQN) relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D3.Global {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D3(shape = shape, value = result)
}

@ScopeOp
infix fun IOType.D3.eq(other: IOType.D3): IOType.D3.Global = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun IOType.D3.eq(
    other: IOType.D3,
    @ScopeOpDefault(EQUALS_ABSOLUTE_TOLERANCE_FQN) absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    @ScopeOpDefault(EQUALS_RELATIVE_TOLERANCE_FQN) relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D3.Global {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D3(shape = shape, value = result)
}

@ScopeOp
infix fun IOType.D3.gt(other: Float): IOType.D2.Global {
    val result = Backend.greaterThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

@ScopeOp
infix fun IOType.D3.gt(other: IOType.D2): IOType.D2.Global {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D2(shape = shape, value = result)
}

@ScopeOp
infix fun IOType.D3.lt(other: Float): IOType.D2.Global {
    val result = Backend.lessThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

@ScopeOp
infix fun IOType.D3.lt(other: IOType.D2): IOType.D2.Global {
    val result = Backend.lessThan(value, other.value)
    return IOType.D2(shape = shape, value = result)
}
