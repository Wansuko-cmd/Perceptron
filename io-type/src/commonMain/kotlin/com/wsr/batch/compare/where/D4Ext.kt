package com.wsr.batch.compare.where

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("WhereFloatToFloatAsD4")
fun where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Float): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("WhereFloatToD4s")
fun where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Batch<IOType.D4>): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("WhereD4sToFloat")
fun where(condition: Batch<IOType.D4>, onTrue: Batch<IOType.D4>, onFalse: Float): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("WhereD4sToD4s")
fun where(condition: Batch<IOType.D4>, onTrue: Batch<IOType.D4>, onFalse: Batch<IOType.D4>): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchFloatWhereFloat")
fun Batch<IOType.D4>.where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Float): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereFloatWithLambda")
inline fun Batch<IOType.D4>.where(
    onTrue: Float,
    onFalse: Float,
    condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
): Batch<IOType.D4> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchFloatWhereD4s")
fun Batch<IOType.D4>.where(
    condition: Batch<IOType.D4>,
    onTrue: Float,
    onFalse: Batch<IOType.D4> = this,
): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereD4sWithLambda")
inline fun Batch<IOType.D4>.where(
    onTrue: Float,
    onFalse: Batch<IOType.D4> = this,
    condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
): Batch<IOType.D4> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD4sWhereFloat")
fun Batch<IOType.D4>.where(
    condition: Batch<IOType.D4>,
    onTrue: Batch<IOType.D4> = this,
    onFalse: Float,
): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD4sWhereFloatWithLambda")
inline fun Batch<IOType.D4>.where(
    onTrue: Batch<IOType.D4> = this,
    onFalse: Float,
    condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD4sWhereD4s")
fun Batch<IOType.D4>.where(
    condition: Batch<IOType.D4>,
    onTrue: Batch<IOType.D4> = this,
    onFalse: Batch<IOType.D4> = this,
): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD4sWhereD4sWithLambda")
inline fun Batch<IOType.D4>.where(
    onTrue: Batch<IOType.D4> = this,
    onFalse: Batch<IOType.D4> = this,
    condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))
