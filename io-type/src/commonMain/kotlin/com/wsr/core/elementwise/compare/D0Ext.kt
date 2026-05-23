package com.wsr.core.elementwise.compare

import com.wsr.Backend
import com.wsr.core.IOType

internal const val EQUALS_ABSOLUTE_TOLERANCE = 1e-4f
internal const val EQUALS_RELATIVE_TOLERANCE = 1e-4f

infix fun IOType.D0.eq(other: Float) = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

fun IOType.D0.eq(
    other: Float,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D0 {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D0(result)
}

infix fun IOType.D0.eq(other: IOType.D0) = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

fun IOType.D0.eq(
    other: IOType.D0,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D0 {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D0(result)
}

infix fun IOType.D0.gt(other: Float): IOType.D0 {
    val result = Backend.greaterThan(value, other)
    return IOType.D0(result)
}

infix fun IOType.D0.gt(other: IOType.D0): IOType.D0 {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D0(result)
}

infix fun IOType.D0.lt(other: Float): IOType.D0 {
    val result = Backend.lessThan(value, other)
    return IOType.D0(result)
}

infix fun IOType.D0.lt(other: IOType.D0): IOType.D0 {
    val result = Backend.lessThan(value, other.value)
    return IOType.D0(result)
}
