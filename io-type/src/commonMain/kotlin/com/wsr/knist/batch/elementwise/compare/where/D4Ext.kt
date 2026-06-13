package com.wsr.knist.batch.elementwise.compare.where

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d4
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchWhereFloatToFloatAsD4")
fun where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Float): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch.d4(condition.size, condition.shape, result)
}

@JvmName("batchWhereFloatToD4s")
fun where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Batch<IOType.D4>): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch.d4(condition.size, condition.shape, result)
}

@JvmName("batchWhereD4sToFloat")
fun where(condition: Batch<IOType.D4>, onTrue: Batch<IOType.D4>, onFalse: Float): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch.d4(condition.size, condition.shape, result)
}

@JvmName("batchWhereD4sToD4s")
fun where(condition: Batch<IOType.D4>, onTrue: Batch<IOType.D4>, onFalse: Batch<IOType.D4>): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch.d4(condition.size, condition.shape, result)
}

@JvmName("batchFloatWhereFloat")
fun Batch<IOType.D4>.where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Float): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch.d4(size, shape, result)
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
    return Batch.d4(size, shape, result)
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
    return Batch.d4(size, shape, result)
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
    return Batch.d4(size, shape, result)
}

@JvmName("batchD4sWhereD4sWithLambda")
inline fun Batch<IOType.D4>.where(
    onTrue: Batch<IOType.D4> = this,
    onFalse: Batch<IOType.D4> = this,
    condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))
