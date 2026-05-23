package com.wsr.core.elementwise.compare.where

import com.wsr.Backend
import com.wsr.core.IOType

fun where(condition: IOType.D3, onTrue: Float, onFalse: Float): IOType.D3 {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return IOType.D3(shape = condition.shape, value = result)
}

fun where(condition: IOType.D3, onTrue: Float, onFalse: IOType.D3): IOType.D3 {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return IOType.D3(shape = condition.shape, value = result)
}

fun where(condition: IOType.D3, onTrue: IOType.D3, onFalse: Float): IOType.D3 {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return IOType.D3(shape = condition.shape, value = result)
}

fun where(condition: IOType.D3, onTrue: IOType.D3, onFalse: IOType.D3): IOType.D3 {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return IOType.D3(shape = condition.shape, value = result)
}

inline fun IOType.D3.where(onTrue: Float, onFalse: Float, condition: (IOType.D3) -> IOType.D3) = where(
    condition = condition(this),
    onTrue = onTrue,
    onFalse = onFalse,
)

fun IOType.D3.where(condition: IOType.D3, onTrue: Float, onFalse: IOType.D3 = this): IOType.D3 {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return IOType.D3(shape = shape, value = result)
}

inline fun IOType.D3.where(onTrue: Float, onFalse: IOType.D3 = this, condition: (IOType.D3) -> IOType.D3) = where(
    condition = condition(this),
    onTrue = onTrue,
    onFalse = onFalse,
)

fun IOType.D3.where(condition: IOType.D3, onTrue: IOType.D3 = this, onFalse: Float): IOType.D3 {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return IOType.D3(shape = shape, value = result)
}

inline fun IOType.D3.where(onTrue: IOType.D3 = this, onFalse: Float, condition: (IOType.D3) -> IOType.D3) = where(
    condition = condition(this),
    onTrue = onTrue,
    onFalse = onFalse,
)

fun IOType.D3.where(condition: IOType.D3, onTrue: IOType.D3 = this, onFalse: IOType.D3 = this): IOType.D3 {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return IOType.D3(shape = shape, value = result)
}

inline fun IOType.D3.where(onTrue: IOType.D3 = this, onFalse: IOType.D3 = this, condition: (IOType.D3) -> IOType.D3) =
    where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )
