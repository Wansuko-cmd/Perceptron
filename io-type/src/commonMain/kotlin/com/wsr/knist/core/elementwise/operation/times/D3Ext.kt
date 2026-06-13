package com.wsr.knist.core.elementwise.operation.times

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
@ScopeOp
operator fun IOType.D3.times(other: Float): IOType.D3 {
    val result = Backend.times(x = value, y = other)
    return IOType.D3.Global(shape = shape, value = result)
}

@ScopeOp
operator fun IOType.D3.times(other: IOType.D0): IOType.D3 {
    val result = Backend.times(x = value, xi = 1, xj = size, y = other.value, axis = 0)
    return IOType.D3.Global(shape = shape, value = result)
}

@ScopeOp
fun IOType.D3.times(other: IOType.D1, axis: Int): IOType.D3 {
    val result = Backend.times(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        y = other.value,
        axis = axis,
    )
    return IOType.D3.Global(shape = shape, value = result)
}

@ScopeOp
fun IOType.D3.times(other: IOType.D2, axis1: Int, axis2: Int): IOType.D3 {
    val result = Backend.times(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        y = other.value,
        yi = other.i,
        yj = other.j,
        axis1 = axis1,
        axis2 = axis2,
    )
    return IOType.D3.Global(shape = shape, value = result)
}

@ScopeOp
operator fun IOType.D3.times(other: IOType.D3): IOType.D3 {
    val result = Backend.times(x = value, y = other.value)
    return IOType.D3.Global(shape = shape, value = result)
}
