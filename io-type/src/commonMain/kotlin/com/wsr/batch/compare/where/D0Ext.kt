package com.wsr.batch.compare.where

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("WhereFloatToD0s")
fun where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("WhereD0sToFloat")
fun where(condition: Batch<IOType.D0>, onTrue: Batch<IOType.D0>, onFalse: Float): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("WhereD0sToD0s")
fun where(condition: Batch<IOType.D0>, onTrue: Batch<IOType.D0>, onFalse: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchFloatWhereD0s")
fun Batch<IOType.D0>.where(
    condition: Batch<IOType.D0>,
    onTrue: Float,
    onFalse: Batch<IOType.D0> = this,
): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereD0sWithLambda")
inline fun Batch<IOType.D0>.where(
    onTrue: Float,
    onFalse: Batch<IOType.D0> = this,
    condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
): Batch<IOType.D0> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD0sWhereFloat")
fun Batch<IOType.D0>.where(
    condition: Batch<IOType.D0>,
    onTrue: Batch<IOType.D0> = this,
    onFalse: Float,
): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sWhereFloatWithLambda")
inline fun Batch<IOType.D0>.where(
    onTrue: Batch<IOType.D0> = this,
    onFalse: Float,
    condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD0sWhereD0s")
fun Batch<IOType.D0>.where(
    condition: Batch<IOType.D0>,
    onTrue: Batch<IOType.D0> = this,
    onFalse: Batch<IOType.D0> = this,
): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sWhereD0sWithLambda")
inline fun Batch<IOType.D0>.where(
    onTrue: Batch<IOType.D0> = this,
    onFalse: Batch<IOType.D0> = this,
    condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))
