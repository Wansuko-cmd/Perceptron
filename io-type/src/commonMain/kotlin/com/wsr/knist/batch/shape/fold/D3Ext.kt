package com.wsr.knist.batch.shape.fold

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d3
import com.wsr.knist.batch.d4
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.batch.l
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp
import kotlin.math.sqrt

@ScopeOp
fun Batch<IOType.D3>.unfold(window: Int, stride: Int, dilation: Int, padding: Int): Batch<IOType.D4.Global> {
    val windowSize = (window - 1) * dilation + 1
    val oj = (j - windowSize + padding * 2) / stride + 1
    val ok = (k - windowSize + padding * 2) / stride + 1
    val result = Backend.unfold(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        b = size,
        window = window,
        stride = stride,
        dilation = dilation,
        padding = padding,
    )
    return Batch.d4(size = size, i = i, j = oj, k = ok, l = window * window, value = result)
}

@ScopeOp
fun Batch<IOType.D4>.fold(stride: Int, dilation: Int, padding: Int): Batch<IOType.D3.Global> {
    val window = sqrt(l.toDouble()).toInt()
    val result = Backend.fold(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        xl = l,
        b = size,
        stride = stride,
        dilation = dilation,
        padding = padding,
    )
    val windowSize = (window - 1) * dilation + 1
    return Batch.d3(
        size = size,
        i = i,
        j = windowSize + (j - 1) * stride - padding * 2,
        k = windowSize + (k - 1) * stride - padding * 2,
        value = result,
    )
}
