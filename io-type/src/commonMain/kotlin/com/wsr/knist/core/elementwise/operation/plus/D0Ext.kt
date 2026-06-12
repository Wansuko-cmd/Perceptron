package com.wsr.knist.core.elementwise.operation.plus

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.unwrap

operator fun Float.plus(other: IOType.D0): IOType.D0 = IOType.d0(value = this + other.unwrap())

operator fun Float.plus(other: IOType.D1): IOType.D1 {
    val result = Backend.plus(x = this, y = other.value)
    return IOType.D1(value = result)
}

operator fun Float.plus(other: IOType.D2): IOType.D2 {
    val result = Backend.plus(x = this, y = other.value)
    return IOType.D2(shape = other.shape, value = result)
}

operator fun Float.plus(other: IOType.D3): IOType.D3 {
    val result = Backend.plus(x = this, y = other.value)
    return IOType.D3(shape = other.shape, value = result)
}

operator fun Float.plus(other: IOType.D4): IOType.D4 {
    val result = Backend.plus(x = this, y = other.value)
    return IOType.D4(shape = other.shape, value = result)
}

operator fun IOType.D0.plus(other: Float): IOType.D0 = IOType.d0(value = unwrap() + other)

operator fun IOType.D0.plus(other: IOType.D0): IOType.D0 = IOType.d0(unwrap() + other.unwrap())

operator fun IOType.D0.plus(other: IOType.D1): IOType.D1 {
    val result = Backend.plus(x = unwrap(), y = other.value)
    return IOType.D1(value = result)
}

operator fun IOType.D0.plus(other: IOType.D2): IOType.D2 {
    val result = Backend.plus(x = unwrap(), y = other.value)
    return IOType.D2(shape = other.shape, value = result)
}

operator fun IOType.D0.plus(other: IOType.D3): IOType.D3 {
    val result = Backend.plus(x = unwrap(), y = other.value)
    return IOType.D3(shape = other.shape, value = result)
}

operator fun IOType.D0.plus(other: IOType.D4): IOType.D4 {
    val result = Backend.plus(x = unwrap(), y = other.value)
    return IOType.D4(shape = other.shape, value = result)
}
