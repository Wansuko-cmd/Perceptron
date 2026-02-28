package com.wsr.core.compare.where

import com.wsr.Backend
import com.wsr.core.IOType

fun where(condition: IOType.D2, onTrue: Float, onFalse: IOType.D2): IOType.D2 {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return IOType.D2(shape = condition.shape, value = result)
}

fun where(condition: IOType.D2, onTrue: IOType.D2, onFalse: Float): IOType.D2 {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return IOType.D2(shape = condition.shape, value = result)
}

fun where(condition: IOType.D2, onTrue: IOType.D2, onFalse: IOType.D2): IOType.D2 {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return IOType.D2(shape = condition.shape, value = result)
}

fun IOType.D2.where(condition: IOType.D2, onTrue: Float, onFalse: IOType.D2 = this): IOType.D2 {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return IOType.D2(shape = shape, value = result)
}

fun IOType.D2.where(condition: IOType.D2, onTrue: IOType.D2 = this, onFalse: Float): IOType.D2 {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return IOType.D2(shape = shape, value = result)
}

fun IOType.D2.where(condition: IOType.D2, onTrue: IOType.D2 = this, onFalse: IOType.D2 = this): IOType.D2 {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return IOType.D2(shape = shape, value = result)
}
