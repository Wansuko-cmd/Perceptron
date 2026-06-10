package com.wsr.knist.batch.elementwise.compare.where

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.jvm.JvmName

@JvmName("batchWhereFloatToFloatAsD3")
fun where(condition: Batch<IOType.D3>, onTrue: Float, onFalse: Float): Batch<IOType.D3> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchWhereFloatToD3s")
fun where(condition: Batch<IOType.D3>, onTrue: Float, onFalse: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchWhereD3sToFloat")
fun where(condition: Batch<IOType.D3>, onTrue: Batch<IOType.D3>, onFalse: Float): Batch<IOType.D3> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchWhereD3sToD3s")
fun where(condition: Batch<IOType.D3>, onTrue: Batch<IOType.D3>, onFalse: Batch<IOType.D3>): Batch<IOType.D3> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchFloatWhereFloat")
@ScopeOp
fun Batch<IOType.D3>.where(condition: Batch<IOType.D3>, onTrue: Float, onFalse: Float): Batch<IOType.D3> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereFloatWithLambda")
@ScopeOp
inline fun Batch<IOType.D3>.where(
    onTrue: Float,
    onFalse: Float,
    condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
): Batch<IOType.D3> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchFloatWhereD3s")
@ScopeOp
fun Batch<IOType.D3>.where(
    condition: Batch<IOType.D3>,
    onTrue: Float,
    onFalse: Batch<IOType.D3> = this,
): Batch<IOType.D3> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchFloatWhereD3sWithLambda")
@ScopeOp
inline fun Batch<IOType.D3>.where(
    onTrue: Float,
    onFalse: Batch<IOType.D3> = this,
    condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
): Batch<IOType.D3> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD3sWhereFloat")
@ScopeOp
fun Batch<IOType.D3>.where(
    condition: Batch<IOType.D3>,
    onTrue: Batch<IOType.D3> = this,
    onFalse: Float,
): Batch<IOType.D3> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sWhereFloatWithLambda")
@ScopeOp
inline fun Batch<IOType.D3>.where(
    onTrue: Batch<IOType.D3> = this,
    onFalse: Float,
    condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD3sWhereD3s")
@ScopeOp
fun Batch<IOType.D3>.where(
    condition: Batch<IOType.D3>,
    onTrue: Batch<IOType.D3> = this,
    onFalse: Batch<IOType.D3> = this,
): Batch<IOType.D3> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sWhereD3sWithLambda")
@ScopeOp
inline fun Batch<IOType.D3>.where(
    onTrue: Batch<IOType.D3> = this,
    onFalse: Batch<IOType.D3> = this,
    condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))
