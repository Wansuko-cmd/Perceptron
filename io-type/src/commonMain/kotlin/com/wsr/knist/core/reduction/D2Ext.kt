package com.wsr.knist.core.reduction

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.D0
import com.wsr.knist.core.D1
import com.wsr.knist.scope.ScopeOp

@ScopeOp
fun IOType.D2.average(): IOType.D0 = IOType.D0(Backend.average(value))

@ScopeOp
fun IOType.D2.average(axis: Int): IOType.D1 {
    val result = Backend.average(x = value, xi = i, xj = j, axis = axis)
    return IOType.D1(value = result)
}

@ScopeOp
fun IOType.D2.max(): IOType.D0 = IOType.D0(Backend.max(x = value))

@ScopeOp
fun IOType.D2.max(axis: Int): IOType.D1 {
    val result = Backend.max(x = value, xi = i, xj = j, axis = axis)
    return IOType.D1(result)
}

@ScopeOp
fun IOType.D2.min() = IOType.D0(Backend.min(x = value))

@ScopeOp
fun IOType.D2.min(axis: Int): IOType.D1 {
    val result = Backend.min(x = value, xi = i, xj = j, axis = axis)
    return IOType.D1(result)
}

@ScopeOp
fun IOType.D2.sum(): IOType.D0 = IOType.D0(Backend.sum(x = value))

@ScopeOp
fun IOType.D2.sum(axis: Int): IOType.D1 {
    val result = Backend.sum(x = value, xi = i, xj = j, axis = axis)
    return IOType.D1(result)
}

@ScopeOp
fun IOType.D2.maxIndex(axis: Int): IOType.D1 {
    val result = Backend.maxIndex(x = value, xi = i, xj = j, axis = axis)
    return IOType.D1(result)
}
