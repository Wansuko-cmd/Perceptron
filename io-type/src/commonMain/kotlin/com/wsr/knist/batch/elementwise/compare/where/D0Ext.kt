package com.wsr.knist.batch.elementwise.compare.where

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d0
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchWhereFloatToFloatAsD0")
fun where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Float): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch.d0(condition.size, result)
}

@JvmName("batchWhereFloatToD0s")
fun where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch.d0(condition.size, result)
}

@JvmName("batchWhereD0sToFloat")
fun where(condition: Batch<IOType.D0>, onTrue: Batch<IOType.D0>, onFalse: Float): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch.d0(condition.size, result)
}

@JvmName("batchWhereD0sToD0s")
fun where(condition: Batch<IOType.D0>, onTrue: Batch<IOType.D0>, onFalse: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch.d0(condition.size, result)
}

@JvmName("batchFloatWhereFloat")
fun Batch<IOType.D0>.where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Float): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch.d0(size, result)
}

@JvmName("batchFloatWhereFloatWithLambda")
inline fun Batch<IOType.D0>.where(
    onTrue: Float,
    onFalse: Float,
    condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
): Batch<IOType.D0> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchFloatWhereD0s")
fun Batch<IOType.D0>.where(
    condition: Batch<IOType.D0>,
    onTrue: Float,
    onFalse: Batch<IOType.D0> = this,
): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch.d0(size, result)
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
    return Batch.d0(size, result)
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
    return Batch.d0(size, result)
}

@JvmName("batchD0sWhereD0sWithLambda")
inline fun Batch<IOType.D0>.where(
    onTrue: Batch<IOType.D0> = this,
    onFalse: Batch<IOType.D0> = this,
    condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))
