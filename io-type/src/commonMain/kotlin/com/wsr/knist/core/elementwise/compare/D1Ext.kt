package com.wsr.knist.core.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType

infix fun IOType.D1.eq(other: Float) = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

fun IOType.D1.eq(
    other: Float,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D1 {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D1(result)
}

infix fun IOType.D1.eq(other: IOType.D1) = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

fun IOType.D1.eq(
    other: IOType.D1,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D1 {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D1(result)
}

infix fun IOType.D1.gt(other: Float): IOType.D1 {
    val result = Backend.greaterThan(value, other)
    return IOType.D1(result)
}

infix fun IOType.D1.gt(other: IOType.D1): IOType.D1 {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D1(result)
}

infix fun IOType.D1.lt(other: Float): IOType.D1 {
    val result = Backend.lessThan(value, other)
    return IOType.D1(result)
}

infix fun IOType.D1.lt(other: IOType.D1): IOType.D1 {
    val result = Backend.lessThan(value, other.value)
    return IOType.D1(result)
}
