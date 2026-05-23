package com.wsr.batch.shape.fold

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.i
import com.wsr.batch.j
import com.wsr.batch.k
import com.wsr.core.IOType

fun Batch<IOType.D2>.unfold(windowSize: Int, stride: Int, padding: Int): Batch<IOType.D3> {
    val result = Backend.unfold(
        x = value,
        xi = i,
        xj = j,
        b = size,
        window = windowSize,
        stride = stride,
        padding = padding,
    )
    return Batch(
        value = result,
        size = size,
        shape = listOf(
            i,
            (j - windowSize + padding * 2) / stride + 1,
            windowSize,
        ),
    )
}

fun Batch<IOType.D3>.fold(stride: Int, padding: Int): Batch<IOType.D2> {
    val result = Backend.fold(
        x = value,
        xi = i,
        xj = j,
        xk = k,
        b = size,
        stride = stride,
        padding = padding,
    )
    return Batch(
        value = result,
        size = size,
        shape = listOf(
            i,
            k + (j - 1) * stride - padding * 2,
        ),
    )
}
