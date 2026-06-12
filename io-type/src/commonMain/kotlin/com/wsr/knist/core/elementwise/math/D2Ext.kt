package com.wsr.knist.core.elementwise.math

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.D2
import com.wsr.knist.core.elementwise.operation.div.div
import com.wsr.knist.core.elementwise.operation.minus.minus
import com.wsr.knist.core.get
import com.wsr.knist.core.reduction.max
import com.wsr.knist.core.reduction.sum
import com.wsr.knist.scope.ScopeOp
import kotlin.math.pow

@ScopeOp
fun IOType.D2.exp(): IOType.D2 {
    val result = Backend.exp(x = value)
    return IOType.D2(shape = shape, value = result)
}

@ScopeOp
fun IOType.D2.ln(e: Float): IOType.D2 {
    val result = Backend.ln(x = value, e = e)
    return IOType.D2(shape = shape, value = result)
}

@ScopeOp
fun IOType.D2.pow(n: Int): IOType.D2 {
    val result = Backend.pow(x = value, n = n)
    return IOType.D2(shape = shape, value = result)
}

@ScopeOp
fun IOType.D2.softmax(): IOType.D2 {
    val max = max()
    val exp = (this - max).exp()
    val sum = exp.sum()
    return exp / sum
}

@ScopeOp
fun IOType.D2.softmax(axis: Int): IOType.D2 {
    val max = max(axis = axis)
    val exp = this.minus(other = max, axis = if (axis == 0) 1 else 0).exp()
    val sum = exp.sum(axis = axis)
    return exp.div(other = sum, axis = if (axis == 0) 1 else 0)
}

@ScopeOp
fun IOType.D2.sqrt(e: Float = 1e-7f): IOType.D2 {
    val result = Backend.sqrt(x = value, e = e)
    return IOType.D2(shape = shape, value = result)
}
