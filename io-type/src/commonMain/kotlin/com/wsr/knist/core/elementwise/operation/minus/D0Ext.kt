package com.wsr.knist.core.elementwise.operation.minus

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType

operator fun Float.minus(other: IOType.D0): IOType.D0 = IOType.D0(Backend.minus(x = this, y = other.value))

operator fun Float.minus(other: IOType.D1): IOType.D1 {
    val result = Backend.minus(x = this, y = other.value)
    return IOType.D1(value = result)
}

operator fun Float.minus(other: IOType.D2): IOType.D2 {
    val result = Backend.minus(x = this, y = other.value)
    return IOType.D2(shape = other.shape, value = result)
}

operator fun Float.minus(other: IOType.D3): IOType.D3 {
    val result = Backend.minus(x = this, y = other.value)
    return IOType.D3(shape = other.shape, value = result)
}

operator fun Float.minus(other: IOType.D4): IOType.D4 {
    val result = Backend.minus(x = this, y = other.value)
    return IOType.D4(shape = other.shape, value = result)
}

operator fun IOType.D0.minus(other: Float): IOType.D0 = IOType.D0(Backend.minus(x = value, y = other))

operator fun IOType.D0.minus(other: IOType.D0): IOType.D0 = IOType.D0(Backend.minus(x = value, y = other.value))

operator fun IOType.D0.minus(other: IOType.D1): IOType.D1 {
    val result = Backend.minus(x = value, y = other.value, yi = 1, yj = other.size, axis = 0)
    return IOType.D1(value = result)
}

operator fun IOType.D0.minus(other: IOType.D2): IOType.D2 {
    val result = Backend.minus(x = value, y = other.value, yi = 1, yj = other.size, axis = 0)
    return IOType.D2(shape = other.shape, value = result)
}

operator fun IOType.D0.minus(other: IOType.D3): IOType.D3 {
    val result = Backend.minus(x = value, y = other.value, yi = 1, yj = other.size, axis = 0)
    return IOType.D3(shape = other.shape, value = result)
}

operator fun IOType.D0.minus(other: IOType.D4): IOType.D4 {
    val result = Backend.minus(x = value, y = other.value, yi = 1, yj = other.size, axis = 0)
    return IOType.D4(shape = other.shape, value = result)
}
