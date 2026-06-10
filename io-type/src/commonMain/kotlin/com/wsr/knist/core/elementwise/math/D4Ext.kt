package com.wsr.knist.core.elementwise.math

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.operation.div.div
import com.wsr.knist.core.elementwise.operation.minus.minus
import com.wsr.knist.core.get
import com.wsr.knist.core.reduction.max
import com.wsr.knist.core.reduction.sum
import kotlin.math.pow
import com.wsr.knist.scope.ScopeOp

@ScopeOp
fun IOType.D4.exp(): IOType.D4 {
    val result = Backend.exp(x = value)
    return IOType.D4(shape = shape, value = result)
}

@ScopeOp
fun IOType.D4.ln(e: Float): IOType.D4 {
    val result = Backend.ln(x = value, e = e)
    return IOType.D4(shape = shape, value = result)
}

@ScopeOp
fun IOType.D4.pow(n: Int): IOType.D4 {
    val result = Backend.pow(x = value, n = n)
    return IOType.D4(shape = shape, value = result)
}

@ScopeOp
fun IOType.D4.softmax(): IOType.D4 {
    val max = max()
    val exp = (this - max).exp()
    val sum = exp.sum()
    return exp / sum
}

@ScopeOp
fun IOType.D4.softmax(axis: Int): IOType.D4 {
    val axis1 = when (axis) {
        0 -> 1
        else -> 0
    }
    val axis2 = when (axis) {
        0, 1 -> 2
        else -> 1
    }
    val axis3 = when (axis) {
        0, 1, 2 -> 3
        else -> 2
    }
    val max = max(axis = axis)
    val exp = this.minus(other = max, axis1 = axis1, axis2 = axis2, axis3 = axis3).exp()
    val sum = exp.sum(axis = axis)
    return exp.div(other = sum, axis1 = axis1, axis2 = axis2, axis3 = axis3)
}

@ScopeOp
fun IOType.D4.sqrt(e: Float = 1e-7f): IOType.D4 {
    val result = Backend.sqrt(x = value, e = e)
    return IOType.D4(shape = shape, value = result)
}
