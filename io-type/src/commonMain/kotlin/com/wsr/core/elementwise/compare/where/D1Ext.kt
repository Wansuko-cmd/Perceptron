package com.wsr.core.elementwise.compare.where

import com.wsr.Backend
import com.wsr.core.IOType

fun where(condition: IOType.D1, onTrue: Float, onFalse: Float): IOType.D1 {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return IOType.D1(result)
}

fun where(condition: IOType.D1, onTrue: Float, onFalse: IOType.D1): IOType.D1 {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return IOType.D1(result)
}

fun where(condition: IOType.D1, onTrue: IOType.D1, onFalse: Float): IOType.D1 {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return IOType.D1(result)
}

fun where(condition: IOType.D1, onTrue: IOType.D1, onFalse: IOType.D1): IOType.D1 {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return IOType.D1(result)
}

inline fun IOType.D1.where(onTrue: Float, onFalse: Float, condition: (IOType.D1) -> IOType.D1) = where(
    condition = condition(this),
    onTrue = onTrue,
    onFalse = onFalse,
)

fun IOType.D1.where(condition: IOType.D1, onTrue: Float, onFalse: IOType.D1 = this): IOType.D1 {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return IOType.D1(result)
}

inline fun IOType.D1.where(onTrue: Float, onFalse: IOType.D1, condition: (IOType.D1) -> IOType.D1) = where(
    condition = condition(this),
    onTrue = onTrue,
    onFalse = onFalse,
)

fun IOType.D1.where(condition: IOType.D1, onTrue: IOType.D1 = this, onFalse: Float): IOType.D1 {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return IOType.D1(result)
}

inline fun IOType.D1.where(onTrue: IOType.D1 = this, onFalse: Float, condition: (IOType.D1) -> IOType.D1) = where(
    condition = condition(this),
    onTrue = onTrue,
    onFalse = onFalse,
)

fun IOType.D1.where(condition: IOType.D1, onTrue: IOType.D1 = this, onFalse: IOType.D1 = this): IOType.D1 {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return IOType.D1(result)
}

inline fun IOType.D1.where(onTrue: IOType.D1 = this, onFalse: IOType.D1, condition: (IOType.D1) -> IOType.D1) = where(
    condition = condition(this),
    onTrue = onTrue,
    onFalse = onFalse,
)
