package com.wsr.knist.core.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.D2
import com.wsr.knist.core.D4
import com.wsr.knist.scope.ScopeOp

@ScopeOp
infix fun IOType.D4.eq(other: Float) = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun IOType.D4.eq(
    other: Float,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D4 {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D4(shape = shape, value = result)
}

@ScopeOp
infix fun IOType.D4.eq(other: IOType.D4) = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@ScopeOp
fun IOType.D4.eq(
    other: IOType.D4,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D4 {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D4(shape = shape, value = result)
}

@ScopeOp
infix fun IOType.D4.gt(other: Float): IOType.D2 {
    val result = Backend.greaterThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

@ScopeOp
infix fun IOType.D4.gt(other: IOType.D2): IOType.D2 {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D2(shape = shape, value = result)
}

@ScopeOp
infix fun IOType.D4.lt(other: Float): IOType.D2 {
    val result = Backend.lessThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

@ScopeOp
infix fun IOType.D4.lt(other: IOType.D2): IOType.D2 {
    val result = Backend.lessThan(value, other.value)
    return IOType.D2(shape = shape, value = result)
}
