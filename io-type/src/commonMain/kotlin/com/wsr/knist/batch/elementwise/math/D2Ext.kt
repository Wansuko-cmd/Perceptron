package com.wsr.knist.batch.elementwise.math

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.shape.toBatch
import com.wsr.knist.batch.shape.toD3
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.math.softmax
import com.wsr.knist.core.shape.reshapeToD2
import com.wsr.knist.core.shape.reshapeToD3
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
fun Batch<IOType.D2>.softmax(): Batch<IOType.D2> = toD3()
    .reshapeToD2(i = size, j = step)
    .softmax(axis = 1)
    .reshapeToD3(i = size, j = i, k = j)
    .toBatch()

@JvmName("batchD2sSoftmaxWithAxis")
fun Batch<IOType.D2>.softmax(axis: Int): Batch<IOType.D2> = toD3()
    .softmax(axis = axis + 1)
    .toBatch()

@JvmName("batchD2sSqrt")
fun Batch<IOType.D2>.sqrt(e: Float = 1e-7f): Batch<IOType.D2> {
    val result = Backend.sqrt(x = value, e = e)
    return Batch(size = size, shape = shape, value = result)
}
