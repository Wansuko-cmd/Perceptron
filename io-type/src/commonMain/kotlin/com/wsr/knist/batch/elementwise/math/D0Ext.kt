package com.wsr.knist.batch.elementwise.math

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d0
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD0sLn")
fun Batch<IOType.D0>.ln(e: Float = 1e-7f): Batch<IOType.D0> {
    val result = Backend.ln(x = value, e = e)
    return Batch.d0(size, result)
}

@JvmName("batchD0sPow")
fun Batch<IOType.D0>.pow(n: Int): Batch<IOType.D0> {
    val result = Backend.pow(x = value, n = n)
    return Batch.d0(size, result)
}

@JvmName("batchD0sSqrt")
fun Batch<IOType.D0>.sqrt(e: Float = 1e-7f): Batch<IOType.D0> {
    val result = Backend.sqrt(x = value, e = e)
    return Batch.d0(size, result)
}
