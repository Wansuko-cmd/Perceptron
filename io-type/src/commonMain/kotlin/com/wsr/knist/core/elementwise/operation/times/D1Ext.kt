package com.wsr.knist.core.elementwise.operation.times

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.unwrap

operator fun IOType.D1.times(other: Float): IOType.D1 {
    val result = Backend.times(x = value, y = other)
    return IOType.D1(value = result)
}

operator fun IOType.D1.times(other: IOType.D0): IOType.D1 {
    val result = Backend.times(x = value, y = other.unwrap())
    return IOType.D1(value = result)
}

operator fun IOType.D1.times(other: IOType.D1): IOType.D1 {
    val result = Backend.times(x = value, y = other.value)
    return IOType.D1(value = result)
}

fun IOType.D1.times(other: IOType.D2, axis: Int): IOType.D2 {
    val result = Backend.times(x = value, y = other.value, yi = other.i, yj = other.j, axis = axis)
    return IOType.D2(shape = other.shape, value = result)
}
