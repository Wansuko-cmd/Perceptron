package com.wsr.knist.batch.shape.fold

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d2
import com.wsr.knist.batch.d3
import com.wsr.knist.batch.i
import com.wsr.knist.batch.j
import com.wsr.knist.batch.k
import com.wsr.knist.core.IOType
import com.wsr.knist.scope.ScopeOp

@ScopeOp
fun Batch<IOType.D2>.unfold(window: Int, stride: Int, dilation: Int, padding: Int): Batch<IOType.D3.Global> {
    val result = Backend.unfold(
        x = value,
        xi = i,
        xj = j,
        b = size,
        window = window,
        stride = stride,
        dilation = dilation,
        padding = padding,
    )
    val windowSize = (window - 1) * dilation + 1
    return Batch.d3(
        size = size,
        i = i,
        j = (j - windowSize + padding * 2) / stride + 1,
        k = window,
        value = result,
    )
}

@ScopeOp
fun Batch<IOType.D3>.fold(stride: Int, dilation: Int, padding: Int): Batch<IOType.D2.Global> {
    val result = Backend.fold(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        b = size,
        stride = stride,
        dilation = dilation,
        padding = padding,
    )
    val windowSize = (k - 1) * dilation + 1
    return Batch.d2(
        size = size,
        i = i,
        j = windowSize + (j - 1) * stride - padding * 2,
        value = result,
    )
}
