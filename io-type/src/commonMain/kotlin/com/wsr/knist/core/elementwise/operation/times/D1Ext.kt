package com.wsr.knist.core.elementwise.operation.times

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
@ScopeOp
operator fun IOType.D1.times(other: Float): IOType.D1 {
    val result = Backend.times(x = value, y = other)
    return IOType.D1(value = result)
}

@ScopeOp
operator fun IOType.D1.times(other: IOType.D0): IOType.D1 {
    val result = Backend.times(x = value, xi = 1, xj = size, y = other.value, axis = 0)
    return IOType.D1(value = result)
}

@ScopeOp
operator fun IOType.D1.times(other: IOType.D1): IOType.D1 {
    val result = Backend.times(x = value, y = other.value)
    return IOType.D1(value = result)
}

@ScopeOp
fun IOType.D1.times(other: IOType.D2, axis: Int): IOType.D2 {
    val result = Backend.times(x = value, y = other.value, yi = other.i, yj = other.j, axis = axis)
    return IOType.D2(shape = other.shape, value = result)
}
