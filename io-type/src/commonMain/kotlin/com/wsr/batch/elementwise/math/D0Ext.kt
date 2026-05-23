package com.wsr.batch.elementwise.math

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD0sLn")
fun Batch<IOType.D0>.ln(e: Float = 1e-7f): Batch<IOType.D0> {
    val result = Backend.ln(x = value, e = e)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sPow")
fun Batch<IOType.D0>.pow(n: Int): Batch<IOType.D0> {
    val result = Backend.pow(x = value, n = n)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD0sSqrt")
fun Batch<IOType.D0>.sqrt(e: Float = 1e-7f): Batch<IOType.D0> {
    val result = Backend.sqrt(x = value, e = e)
    return Batch(size = size, shape = shape, value = result)
}
