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
fun Batch<IOType.D2>.unfold(windowSize: Int, stride: Int, padding: Int): Batch<IOType.D3.Global> {
    val result = Backend.unfold(
        x = value,
        xi = i,
        xj = j,
        b = size,
        window = windowSize,
        stride = stride,
        padding = padding,
    )
    return Batch.d3(
        size,
        i,
        (j - windowSize + padding * 2) / stride + 1,
        windowSize,
        result,
    )
}

@ScopeOp
fun Batch<IOType.D3>.fold(stride: Int, padding: Int): Batch<IOType.D2.Global> {
    val result = Backend.fold(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        b = size,
        stride = stride,
        padding = padding,
    )
    return Batch.d2(
        size,
        i,
        k + (j - 1) * stride - padding * 2,
        result,
    )
}
