package com.wsr.batch.compare.where

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("WhereFloatToFloatAsD1")
fun where(condition: Batch<IOType.D1>, onTrue: Float, onFalse: Float): Batch<IOType.D1> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("WhereFloatToD1s")
fun where(condition: Batch<IOType.D1>, onTrue: Float, onFalse: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("WhereD1sToFloat")
fun where(condition: Batch<IOType.D1>, onTrue: Batch<IOType.D1>, onFalse: Float): Batch<IOType.D1> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("WhereD1sToD1s")
fun where(condition: Batch<IOType.D1>, onTrue: Batch<IOType.D1>, onFalse: Batch<IOType.D1>): Batch<IOType.D1> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchFloatWhereFloat")
fun Batch<IOType.D1>.where(condition: Batch<IOType.D1>, onTrue: Float, onFalse: Float): Batch<IOType.D1> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereFloatWithLambda")
inline fun Batch<IOType.D1>.where(
    onTrue: Float,
    onFalse: Float,
    condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
): Batch<IOType.D1> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchFloatWhereD1s")
fun Batch<IOType.D1>.where(
    condition: Batch<IOType.D1>,
    onTrue: Float,
    onFalse: Batch<IOType.D1> = this,
): Batch<IOType.D1> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereD1sWithLambda")
inline fun Batch<IOType.D1>.where(
    onTrue: Float,
    onFalse: Batch<IOType.D1> = this,
    condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
): Batch<IOType.D1> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD1sWhereFloat")
fun Batch<IOType.D1>.where(
    condition: Batch<IOType.D1>,
    onTrue: Batch<IOType.D1> = this,
    onFalse: Float,
): Batch<IOType.D1> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sWhereFloatWithLambda")
inline fun Batch<IOType.D1>.where(
    onTrue: Batch<IOType.D1> = this,
    onFalse: Float,
    condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD1sWhereD1s")
fun Batch<IOType.D1>.where(
    condition: Batch<IOType.D1>,
    onTrue: Batch<IOType.D1> = this,
    onFalse: Batch<IOType.D1> = this,
): Batch<IOType.D1> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sWhereD1sWithLambda")
inline fun Batch<IOType.D1>.where(
    onTrue: Batch<IOType.D1> = this,
    onFalse: Batch<IOType.D1> = this,
    condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))
