package com.wsr.knist.batch.elementwise.math

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d0
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import com.wsr.knist.scope.ScopeOpDefault
import kotlin.jvm.JvmName

@JvmName("batchD0sLn")
@ScopeOp
fun Batch<IOType.D0>.ln(@ScopeOpDefault("1e-7f") e: Float = 1e-7f): Batch<IOType.D0> {
    val result = Backend.ln(x = value, e = e)
    return Batch.d0(size, result)
}

@JvmName("batchD0sPow")
@ScopeOp
fun Batch<IOType.D0>.pow(n: Int): Batch<IOType.D0.Global> {
    val result = Backend.pow(x = value, n = n)
    return Batch.d0(size, result)
}

@JvmName("batchD0sSqrt")
@ScopeOp
fun Batch<IOType.D0>.sqrt(@ScopeOpDefault("1e-7f") e: Float = 1e-7f): Batch<IOType.D0.Global> {
    val result = Backend.sqrt(x = value, e = e)
    return Batch.d0(size, result)
}
