package com.wsr.batch.math

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.collecction.map.mapValue
import com.wsr.core.IOType
import kotlin.jvm.JvmName
import kotlin.math.exp

@JvmName("batchD1sSigmoid")
fun Batch<IOType.D1>.sigmoid(): Batch<IOType.D1> = Batch(size = size, shape = shape, value = Backend.sigmoid(value))

@JvmName("batchD2sSigmoid")
fun Batch<IOType.D2>.sigmoid(): Batch<IOType.D2> = Batch(size = size, shape = shape, value = Backend.sigmoid(value))

@JvmName("batchD3sSigmoid")
fun Batch<IOType.D3>.sigmoid(): Batch<IOType.D3> = Batch(size = size, shape = shape, value = Backend.sigmoid(value))
