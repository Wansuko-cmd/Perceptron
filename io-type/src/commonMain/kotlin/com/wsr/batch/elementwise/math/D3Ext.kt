package com.wsr.batch.elementwise.math

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.shape.toBatch
import com.wsr.batch.shape.toD4
import com.wsr.core.IOType
import com.wsr.core.elementwise.math.softmax
import com.wsr.core.shape.reshapeToD2
import com.wsr.core.shape.reshapeToD4
import kotlin.jvm.JvmName

@JvmName("batchD3sExp")
fun Batch<IOType.D3>.exp(): Batch<IOType.D3> {
    val result = Backend.exp(x = value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sLn")
fun Batch<IOType.D3>.ln(e: Float = 1e-7f): Batch<IOType.D3> {
    val result = Backend.ln(x = value, e = e)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sPow")
fun Batch<IOType.D3>.pow(n: Int): Batch<IOType.D3> {
    val result = Backend.pow(x = value, n = n)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD3sSigmoid")
fun Batch<IOType.D3>.sigmoid(): Batch<IOType.D3> = Batch(size = size, shape = shape, value = Backend.sigmoid(value))

@JvmName("batchD3sSoftmax")
fun Batch<IOType.D3>.softmax(): Batch<IOType.D3> = toD4()
    .reshapeToD2(i = size, j = step)
    .softmax(axis = 1)
    .reshapeToD4(i = size, j = shape[0], k = shape[1], l = shape[2])
    .toBatch()

@JvmName("batchD3sSoftmaxWithAxis")
fun Batch<IOType.D3>.softmax(axis: Int): Batch<IOType.D3> = toD4()
    .softmax(axis = axis + 1)
    .toBatch()

@JvmName("batchD3sSqrt")
fun Batch<IOType.D3>.sqrt(e: Float = 1e-7f): Batch<IOType.D3> {
    val result = Backend.sqrt(x = value, e = e)
    return Batch(size = size, shape = shape, value = result)
}
