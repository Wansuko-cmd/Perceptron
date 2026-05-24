package com.wsr.batch.shape.fold

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.batch.i
import com.wsr.batch.j
import com.wsr.batch.k
import com.wsr.batch.l
import com.wsr.core.IOType
import kotlin.math.sqrt

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
