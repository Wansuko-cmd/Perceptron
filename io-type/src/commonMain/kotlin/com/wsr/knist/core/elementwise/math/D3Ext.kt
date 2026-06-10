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
fun IOType.D3.exp(): IOType.D3 {
    val result = Backend.exp(x = value)
    return IOType.D3(shape = shape, value = result)
}

@ScopeOp
fun IOType.D3.ln(e: Float): IOType.D3 {
    val result = Backend.ln(x = value, e = e)
    return IOType.D3(shape = shape, value = result)
}

@ScopeOp
fun IOType.D3.pow(n: Int): IOType.D3 {
    val result = Backend.pow(x = value, n = n)
    return IOType.D3(shape = shape, value = result)
}

@ScopeOp
fun IOType.D3.softmax(): IOType.D3 {
    val max = max()
    val exp = (this - max).exp()
    val sum = exp.sum()
    return exp / sum
}

@ScopeOp
fun IOType.D3.softmax(axis: Int): IOType.D3 {
    val axis1 = when (axis) {
        0 -> 1
        else -> 0
    }
    val axis2 = when (axis) {
        0, 1 -> 2
        else -> 1
    }
    val max = max(axis = axis)
    val exp = this.minus(other = max, axis1 = axis1, axis2 = axis2).exp()
    val sum = exp.sum(axis = axis)
    return exp.div(other = sum, axis1 = axis1, axis2 = axis2)
}

@ScopeOp
fun IOType.D3.sqrt(e: Float = 1e-7f): IOType.D3 {
    val result = Backend.sqrt(x = value, e = e)
    return IOType.D3(shape = shape, value = result)
}
