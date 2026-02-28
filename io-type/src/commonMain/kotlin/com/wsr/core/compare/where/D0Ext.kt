package com.wsr.core.compare.where

import com.wsr.Backend
import com.wsr.core.IOType

fun where(condition: IOType.D0, onTrue: Float, onFalse: IOType.D0): IOType.D0 {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return IOType.D0(result)
}

fun where(condition: IOType.D0, onTrue: IOType.D0, onFalse: Float): IOType.D0 {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return IOType.D0(result)
}

fun where(condition: IOType.D0, onTrue: IOType.D0, onFalse: IOType.D0): IOType.D0 {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return IOType.D0(result)
}

fun IOType.D0.where(condition: IOType.D0, onTrue: Float, onFalse: IOType.D0 = this): IOType.D0 {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return IOType.D0(result)
}

fun IOType.D0.where(condition: IOType.D0, onTrue: IOType.D0 = this, onFalse: Float): IOType.D0 {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return IOType.D0(result)
}

fun IOType.D0.where(condition: IOType.D0, onTrue: IOType.D0 = this, onFalse: IOType.D0 = this): IOType.D0 {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return IOType.D0(result)
}
