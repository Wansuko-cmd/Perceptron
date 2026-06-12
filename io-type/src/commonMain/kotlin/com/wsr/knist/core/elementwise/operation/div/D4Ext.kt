package com.wsr.knist.core.elementwise.operation.div

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.unwrap

operator fun IOType.D4.div(other: Float): IOType.D4 {
    val result = Backend.div(x = value, y = other)
    return IOType.D4(shape = shape, value = result)
}

operator fun IOType.D4.div(other: IOType.D0): IOType.D4 {
    val result = Backend.div(x = value, y = other.unwrap())
    return IOType.D4(shape = shape, value = result)
}

fun IOType.D4.div(other: IOType.D1, axis: Int): IOType.D4 {
    val result = Backend.div(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        xl = l,
        y = other.value,
        axis = axis,
    )
    return IOType.D4(shape = shape, value = result)
}

fun IOType.D4.div(other: IOType.D2, axis1: Int, axis2: Int): IOType.D4 {
    val result = Backend.div(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        xl = l,
        y = other.value,
        yi = other.i,
        yj = other.j,
        axis1 = axis1,
        axis2 = axis2,
    )
    return IOType.D4(shape = shape, value = result)
}

fun IOType.D4.div(other: IOType.D3, axis1: Int, axis2: Int, axis3: Int): IOType.D4 {
    val result = Backend.div(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        xl = l,
        y = other.value,
        yi = other.i,
        yj = other.j,
        yk = other.k,
        axis1 = axis1,
        axis2 = axis2,
        axis3 = axis3,
    )
    return IOType.D4(shape = shape, value = result)
}

operator fun IOType.D4.div(other: IOType.D4): IOType.D4 {
    val result = Backend.div(x = value, y = other.value)
    return IOType.D4(shape = shape, value = result)
}
