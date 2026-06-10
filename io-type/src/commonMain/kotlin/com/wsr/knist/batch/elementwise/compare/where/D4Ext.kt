package com.wsr.knist.batch.elementwise.compare.where

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("batchWhereFloatToFloatAsD4")
fun where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Float): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchWhereFloatToD4s")
fun where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Batch<IOType.D4>): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchWhereD4sToFloat")
fun where(condition: Batch<IOType.D4>, onTrue: Batch<IOType.D4>, onFalse: Float): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchWhereD4sToD4s")
fun where(condition: Batch<IOType.D4>, onTrue: Batch<IOType.D4>, onFalse: Batch<IOType.D4>): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchFloatWhereFloat")
@ScopeOp
fun Batch<IOType.D4>.where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Float): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereFloatWithLambda")
@ScopeOp
inline fun Batch<IOType.D4>.where(
    onTrue: Float,
    onFalse: Float,
    condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
): Batch<IOType.D4> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchFloatWhereD4s")
@ScopeOp
fun Batch<IOType.D4>.where(
    condition: Batch<IOType.D4>,
    onTrue: Float,
    onFalse: Batch<IOType.D4> = this,
): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereD4sWithLambda")
@ScopeOp
inline fun Batch<IOType.D4>.where(
    onTrue: Float,
    onFalse: Batch<IOType.D4> = this,
    condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
): Batch<IOType.D4> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD4sWhereFloat")
@ScopeOp
fun Batch<IOType.D4>.where(
    condition: Batch<IOType.D4>,
    onTrue: Batch<IOType.D4> = this,
    onFalse: Float,
): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD4sWhereFloatWithLambda")
@ScopeOp
inline fun Batch<IOType.D4>.where(
    onTrue: Batch<IOType.D4> = this,
    onFalse: Float,
    condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD4sWhereD4s")
@ScopeOp
fun Batch<IOType.D4>.where(
    condition: Batch<IOType.D4>,
    onTrue: Batch<IOType.D4> = this,
    onFalse: Batch<IOType.D4> = this,
): Batch<IOType.D4> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD4sWhereD4sWithLambda")
@ScopeOp
inline fun Batch<IOType.D4>.where(
    onTrue: Batch<IOType.D4> = this,
    onFalse: Batch<IOType.D4> = this,
    condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))
