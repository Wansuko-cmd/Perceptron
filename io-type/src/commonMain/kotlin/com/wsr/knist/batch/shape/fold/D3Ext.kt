package com.wsr.knist.batch.shape.fold

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.batch.l
import com.wsr.knist.core.IOType
import kotlin.math.sqrt
import com.wsr.knist.scope.ScopeOp

@ScopeOp
fun Batch<IOType.D3>.unfold(windowSize: Int, stride: Int, padding: Int): Batch<IOType.D4> {
    val oj = (j - windowSize + padding * 2) / stride + 1
    val ok = (k - windowSize + padding * 2) / stride + 1
    val result = Backend.unfold(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        b = size,
        window = windowSize,
        stride = stride,
        padding = padding,
    )
    return Batch(
        value = result,
        size = size,
        shape = listOf(i, oj, ok, windowSize * windowSize),
    )
}

@ScopeOp
fun Batch<IOType.D4>.fold(stride: Int, padding: Int): Batch<IOType.D3> {
    val window = sqrt(l.toDouble()).toInt()
    val result = Backend.fold(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        xl = l,
        b = size,
        stride = stride,
        padding = padding,
    )
    return Batch(
        value = result,
        size = size,
        shape = listOf(
            i,
            window + (j - 1) * stride - padding * 2,
            window + (k - 1) * stride - padding * 2,
        ),
    )
}
