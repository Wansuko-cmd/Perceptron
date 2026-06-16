package com.wsr.knist.core.elementwise.operation.minus

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
@ScopeOp
operator fun IOType.D2.minus(other: Float): IOType.D2.Global {
    val result = Backend.minus(x = value, y = other)
    return IOType.D2.Global(shape = shape, value = result)
}

@ScopeOp
operator fun IOType.D2.minus(other: IOType.D0): IOType.D2.Global {
    val result = Backend.minus(x = value, xi = 1, xj = size, y = other.value, axis = 0)
    return IOType.D2.Global(shape = shape, value = result)
}

@ScopeOp
fun IOType.D2.minus(other: IOType.D1, axis: Int): IOType.D2.Global {
    val result = Backend.minus(x = value, xi = i, xj = j, y = other.value, axis = axis)
    return IOType.D2.Global(shape = shape, value = result)
}

@ScopeOp
operator fun IOType.D2.minus(other: IOType.D2): IOType.D2.Global {
    val result = Backend.minus(x = value, y = other.value)
    return IOType.D2.Global(shape = shape, value = result)
}

@ScopeOp
fun IOType.D2.minus(other: IOType.D3, axis1: Int, axis2: Int): IOType.D3.Global {
    val result = Backend.minus(
        x = value,
        xi = i,
        xj = j,
        y = other.value,
        yi = other.i,
        yj = other.j,
        yk = other.k,
        axis1 = axis1,
        axis2 = axis2,
    )
    return IOType.D3.Global(shape = other.shape, value = result)
}
