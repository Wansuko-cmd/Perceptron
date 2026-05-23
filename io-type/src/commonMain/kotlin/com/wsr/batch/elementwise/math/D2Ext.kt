package com.wsr.batch.elementwise.math

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.elementwise.map.map
import com.wsr.core.IOType
import com.wsr.core.elementwise.math.softmax
import kotlin.jvm.JvmName

@JvmName("batchD2sExp")
fun Batch<IOType.D2>.exp(): Batch<IOType.D2> {
    val result = Backend.exp(x = value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sLn")
fun Batch<IOType.D2>.ln(e: Float = 1e-7f): Batch<IOType.D2> {
    val result = Backend.ln(x = value, e = e)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sPow")
fun Batch<IOType.D2>.pow(n: Int): Batch<IOType.D2> {
    val result = Backend.pow(x = value, n = n)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD2sSigmoid")
fun Batch<IOType.D2>.sigmoid(): Batch<IOType.D2> = Batch(size = size, shape = shape, value = Backend.sigmoid(value))

@JvmName("batchD2sSoftmax")
fun Batch<IOType.D2>.softmax(): Batch<IOType.D2> = map { it.softmax() }

@JvmName("batchD2sSoftmaxWithAxis")
fun Batch<IOType.D2>.softmax(axis: Int): Batch<IOType.D2> = map { it.softmax(axis = axis) }

@JvmName("batchD2sSqrt")
fun Batch<IOType.D2>.sqrt(e: Float = 1e-7f): Batch<IOType.D2> {
    val result = Backend.sqrt(x = value, e = e)
    return Batch(size = size, shape = shape, value = result)
}
