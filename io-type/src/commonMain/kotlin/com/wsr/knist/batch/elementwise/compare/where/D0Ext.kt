package com.wsr.knist.batch.elementwise.compare.where

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("batchWhereFloatToFloatAsD0")
fun where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Float): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchWhereFloatToD0s")
fun where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchWhereD0sToFloat")
fun where(condition: Batch<IOType.D0>, onTrue: Batch<IOType.D0>, onFalse: Float): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchWhereD0sToD0s")
fun where(condition: Batch<IOType.D0>, onTrue: Batch<IOType.D0>, onFalse: Batch<IOType.D0>): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchFloatWhereFloat")
@ScopeOp
fun Batch<IOType.D0>.where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Float): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereFloatWithLambda")
@ScopeOp
inline fun Batch<IOType.D0>.where(
    onTrue: Float,
    onFalse: Float,
    condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
): Batch<IOType.D0> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchFloatWhereD0s")
@ScopeOp
fun Batch<IOType.D0>.where(
    condition: Batch<IOType.D0>,
    onTrue: Float,
    onFalse: Batch<IOType.D0> = this,
): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereD0sWithLambda")
@ScopeOp
inline fun Batch<IOType.D0>.where(
    onTrue: Float,
    onFalse: Batch<IOType.D0> = this,
    condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
): Batch<IOType.D0> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD0sWhereFloat")
@ScopeOp
fun Batch<IOType.D0>.where(
    condition: Batch<IOType.D0>,
    onTrue: Batch<IOType.D0> = this,
    onFalse: Float,
): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sWhereFloatWithLambda")
@ScopeOp
inline fun Batch<IOType.D0>.where(
    onTrue: Batch<IOType.D0> = this,
    onFalse: Float,
    condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD0sWhereD0s")
@ScopeOp
fun Batch<IOType.D0>.where(
    condition: Batch<IOType.D0>,
    onTrue: Batch<IOType.D0> = this,
    onFalse: Batch<IOType.D0> = this,
): Batch<IOType.D0> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sWhereD0sWithLambda")
@ScopeOp
inline fun Batch<IOType.D0>.where(
    onTrue: Batch<IOType.D0> = this,
    onFalse: Batch<IOType.D0> = this,
    condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))
