package com.wsr.batch.elementwise.math

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.elementwise.map.map
import com.wsr.core.IOType
import com.wsr.core.elementwise.math.softmax
import kotlin.jvm.JvmName

@JvmName("batchD1sExp")
fun Batch<IOType.D1>.exp(): Batch<IOType.D1> {
    val result = Backend.exp(x = value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sLn")
fun Batch<IOType.D1>.ln(e: Float = 1e-7f): Batch<IOType.D1> {
    val result = Backend.ln(x = value, e = e)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sPow")
fun Batch<IOType.D1>.pow(n: Int): Batch<IOType.D1> {
    val result = Backend.pow(x = value, n = n)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD1sSigmoid")
fun Batch<IOType.D1>.sigmoid(): Batch<IOType.D1> = Batch(size = size, shape = shape, value = Backend.sigmoid(value))

@JvmName("batchD1sSoftmax")
fun Batch<IOType.D1>.softmax(): Batch<IOType.D1> = map { it.softmax() }

@JvmName("batchD1sSqrt")
fun Batch<IOType.D1>.sqrt(e: Float = 1e-7f): Batch<IOType.D1> {
    val result = Backend.sqrt(x = value, e = e)
    return Batch(size = size, shape = shape, value = result)
}
