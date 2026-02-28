package com.wsr.core.compare.less

import com.wsr.Backend
import com.wsr.core.IOType

infix fun IOType.D0.lt(other: Float): IOType.D0 {
    val result = Backend.lessThan(value, other)
    return IOType.D0(result)
}

infix fun IOType.D0.lt(other: IOType.D0): IOType.D0 {
    val result = Backend.lessThan(value, other.value)
    return IOType.D0(result)
}

infix fun IOType.D1.lt(other: Float): IOType.D1 {
    val result = Backend.lessThan(value, other)
    return IOType.D1(result)
}

infix fun IOType.D1.lt(other: IOType.D1): IOType.D1 {
    val result = Backend.lessThan(value, other.value)
    return IOType.D1(result)
}

infix fun IOType.D2.lt(other: Float): IOType.D2 {
    val result = Backend.lessThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

infix fun IOType.D2.lt(other: IOType.D2): IOType.D2 {
    val result = Backend.lessThan(value, other.value)
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

infix fun IOType.D4.lt(other: Float): IOType.D2 {
    val result = Backend.lessThan(value, other)
    return IOType.D2(shape = shape, value = result)
}

infix fun IOType.D4.lt(other: IOType.D2): IOType.D2 {
    val result = Backend.lessThan(value, other.value)
    return IOType.D2(shape = shape, value = result)
}
