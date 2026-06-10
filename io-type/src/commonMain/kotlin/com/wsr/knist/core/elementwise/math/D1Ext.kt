package com.wsr.knist.core.elementwise.math

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.operation.div.div
import com.wsr.knist.core.elementwise.operation.minus.minus
import com.wsr.knist.core.get
import com.wsr.knist.core.reduction.max
import com.wsr.knist.core.reduction.sum
import com.wsr.knist.scope.ScopeOp
import kotlin.math.pow

@ScopeOp
fun IOType.D1.exp(): IOType.D1 {
    val result = Backend.exp(x = value)
    return IOType.D1(value = result)
}

@ScopeOp
fun IOType.D1.ln(e: Float): IOType.D1 {
    val result = Backend.ln(x = value, e = e)
    return IOType.D1(value = result)
}

@ScopeOp
fun IOType.D1.pow(n: Int): IOType.D1 {
    val result = Backend.pow(x = value, n = n)
    return IOType.D1(value = result)
}

@ScopeOp
fun IOType.D1.softmax(): IOType.D1 {
    val max = max()
    val exp = (this - max).exp()
    val sum = exp.sum()
    return exp / sum
}

@ScopeOp
fun IOType.D1.sqrt(e: Float = 1e-7f): IOType.D1 {
    val result = Backend.sqrt(x = value, e = e)
    return IOType.D1(value = result)
}
