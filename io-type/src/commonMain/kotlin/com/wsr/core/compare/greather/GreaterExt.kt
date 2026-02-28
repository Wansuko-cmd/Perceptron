package com.wsr.core.compare.greather

import com.wsr.Backend
import com.wsr.core.IOType

infix fun IOType.D0.gt(other: Float): IOType.D0 {
    val result = Backend.greaterThan(value, other)
    return IOType.D0(result)
}

infix fun IOType.D0.gt(other: IOType.D0): IOType.D0 {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D0(result)
}

infix fun IOType.D1.gt(other: Float): IOType.D1 {
    val result = Backend.greaterThan(value, other)
    return IOType.D1(result)
}

infix fun IOType.D1.gt(other: IOType.D1): IOType.D1 {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D1(result)
}

infix fun IOType.D2.gt(other: Float): IOType.D2 {
    val result = Backend.greaterThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

infix fun IOType.D2.gt(other: IOType.D2): IOType.D2 {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D2(shape = shape, value = result)
}

infix fun IOType.D3.gt(other: Float): IOType.D2 {
    val result = Backend.greaterThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

infix fun IOType.D3.gt(other: IOType.D2): IOType.D2 {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D2(shape = shape, value = result)
}

infix fun IOType.D4.gt(other: Float): IOType.D2 {
    val result = Backend.greaterThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

infix fun IOType.D4.gt(other: IOType.D2): IOType.D2 {
    val result = Backend.greaterThan(value, other.value)
    return IOType.D2(shape = shape, value = result)
}
