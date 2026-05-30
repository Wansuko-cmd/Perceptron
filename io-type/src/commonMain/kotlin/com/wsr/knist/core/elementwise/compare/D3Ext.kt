package com.wsr.knist.core.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType

infix fun IOType.D3.eq(other: Float) = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

fun IOType.D3.eq(
    other: Float,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D3 {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D3(shape = shape, value = result)
}

infix fun IOType.D3.eq(other: IOType.D3) = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

fun IOType.D3.eq(
    other: IOType.D3,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): IOType.D3 {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return IOType.D3(shape = shape, value = result)
}

infix fun IOType.D3.gt(other: Float): IOType.D2 {
    val result = Backend.greaterThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

infix fun IOType.D3.gt(other: IOType.D2): IOType.D2 {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D2(shape = shape, value = result)
}

infix fun IOType.D3.lt(other: Float): IOType.D2 {
    val result = Backend.lessThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

infix fun IOType.D3.lt(other: IOType.D2): IOType.D2 {
    val result = Backend.lessThan(value, other.value)
    return IOType.D2(shape = shape, value = result)
}
