package com.wsr.knist.core.elementwise.operation.minus

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp

@ScopeOp
operator fun Float.minus(other: IOType.D0): IOType.D0 = IOType.D0.Global(Backend.minus(x = this, y = other.value))

@ScopeOp
operator fun Float.minus(other: IOType.D1): IOType.D1.Global {
    val result = Backend.minus(x = this, y = other.value)
    return IOType.D1.Global(value = result)
}

@ScopeOp
operator fun Float.minus(other: IOType.D2): IOType.D2.Global {
    val result = Backend.minus(x = this, y = other.value)
    return IOType.D2.Global(shape = other.shape, value = result)
}

@ScopeOp
operator fun Float.minus(other: IOType.D3): IOType.D3.Global {
    val result = Backend.minus(x = this, y = other.value)
    return IOType.D3.Global(shape = other.shape, value = result)
}

@ScopeOp
operator fun Float.minus(other: IOType.D4): IOType.D4.Global {
    val result = Backend.minus(x = this, y = other.value)
    return IOType.D4.Global(shape = other.shape, value = result)
}

@ScopeOp
operator fun IOType.D0.minus(other: Float): IOType.D0 = IOType.D0.Global(Backend.minus(x = value, y = other))

@ScopeOp
operator fun IOType.D0.minus(other: IOType.D0): IOType.D0 = IOType.D0.Global(Backend.minus(x = value, y = other.value))

@ScopeOp
operator fun IOType.D0.minus(other: IOType.D1): IOType.D1.Global {
    val result = Backend.minus(x = value, y = other.value, yi = 1, yj = other.size, axis = 0)
    return IOType.D1.Global(value = result)
}

@ScopeOp
operator fun IOType.D0.minus(other: IOType.D2): IOType.D2.Global {
    val result = Backend.minus(x = value, y = other.value, yi = 1, yj = other.size, axis = 0)
    return IOType.D2.Global(shape = other.shape, value = result)
}

@ScopeOp
operator fun IOType.D0.minus(other: IOType.D3): IOType.D3.Global {
    val result = Backend.minus(x = value, y = other.value, yi = 1, yj = other.size, axis = 0)
    return IOType.D3.Global(shape = other.shape, value = result)
}

@ScopeOp
operator fun IOType.D0.minus(other: IOType.D4): IOType.D4.Global {
    val result = Backend.minus(x = value, y = other.value, yi = 1, yj = other.size, axis = 0)
    return IOType.D4.Global(shape = other.shape, value = result)
}
