package com.wsr.batch.compare.where

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("WhereFloatToD2s")
fun where(condition: Batch<IOType.D2>, onTrue: Float, onFalse: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("WhereD2sToFloat")
fun where(condition: Batch<IOType.D2>, onTrue: Batch<IOType.D2>, onFalse: Float): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("WhereD2sToD2s")
fun where(condition: Batch<IOType.D2>, onTrue: Batch<IOType.D2>, onFalse: Batch<IOType.D2>): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = condition.size, shape = condition.shape, value = result)
}

@JvmName("batchFloatWhereD2s")
fun Batch<IOType.D2>.where(
    condition: Batch<IOType.D2>,
    onTrue: Float,
    onFalse: Batch<IOType.D2> = this,
): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sWhereFloat")
fun Batch<IOType.D2>.where(
    condition: Batch<IOType.D2>,
    onTrue: Batch<IOType.D2> = this,
    onFalse: Float,
): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue.value, onFalse)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sWhereD2s")
fun Batch<IOType.D2>.where(
    condition: Batch<IOType.D2>,
    onTrue: Batch<IOType.D2> = this,
    onFalse: Batch<IOType.D2> = this,
): Batch<IOType.D2> {
    val result = Backend.where(condition.value, onTrue.value, onFalse.value)
    return Batch(size = size, shape = shape, value = result)
}
