package com.wsr.knist.batch.elementwise.compare.where

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d2
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchWhereFloatToFloatAsD2")
fun where(condition: Batch<IOType.D2>, onTrue: Float, onFalse: Float): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch.d2(condition.size, condition.shape, result)
}

@JvmName("batchWhereFloatToD2s")
fun where(condition: Batch<IOType.D2>, onTrue: Float, onFalse: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch.d2(condition.size, condition.shape, result)
}

@JvmName("batchWhereD2sToFloat")
fun where(condition: Batch<IOType.D2>, onTrue: Batch<IOType.D2>, onFalse: Float): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch.d2(condition.size, condition.shape, result)
}

@JvmName("batchWhereD2sToD2s")
fun where(condition: Batch<IOType.D2>, onTrue: Batch<IOType.D2>, onFalse: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch.d2(condition.size, condition.shape, result)
}

@JvmName("batchFloatWhereFloat")
fun Batch<IOType.D2>.where(condition: Batch<IOType.D2>, onTrue: Float, onFalse: Float): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue, onFalse)
    return Batch.d2(size, shape, result)
}

@JvmName("batchFloatWhereFloatWithLambda")
inline fun Batch<IOType.D2>.where(
    onTrue: Float,
    onFalse: Float,
    condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
): Batch<IOType.D2> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchFloatWhereD2s")
fun Batch<IOType.D2>.where(
    condition: Batch<IOType.D2>,
    onTrue: Float,
    onFalse: Batch<IOType.D2> = this,
): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch.d2(size, shape, result)
}

@JvmName("batchFloatWhereD2sWithLambda")
inline fun Batch<IOType.D2>.where(
    onTrue: Float,
    onFalse: Batch<IOType.D2> = this,
    condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
): Batch<IOType.D2> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD2sWhereFloat")
fun Batch<IOType.D2>.where(
    condition: Batch<IOType.D2>,
    onTrue: Batch<IOType.D2> = this,
    onFalse: Float,
): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sWhereFloatWithLambda")
inline fun Batch<IOType.D2>.where(
    onTrue: Batch<IOType.D2> = this,
    onFalse: Float,
    condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

@JvmName("batchD2sWhereD2s")
fun Batch<IOType.D2>.where(
    condition: Batch<IOType.D2>,
    onTrue: Batch<IOType.D2> = this,
    onFalse: Batch<IOType.D2> = this,
): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch.d2(size, shape, result)
}

@JvmName("batchD2sWhereD2sWithLambda")
inline fun Batch<IOType.D2>.where(
    onTrue: Batch<IOType.D2> = this,
    onFalse: Batch<IOType.D2> = this,
    condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))
