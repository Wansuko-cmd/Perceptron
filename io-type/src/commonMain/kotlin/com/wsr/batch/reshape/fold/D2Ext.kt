package com.wsr.batch.reshape.fold

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType

fun Batch<IOType.D2>.unfold(windowSize: Int, stride: Int, padding: Int): Batch<IOType.D3> {
    val result = Backend.unfold(
        x = value,
        xi = shape[0],
        xj = shape[1],
        b = size,
        window = windowSize,
        stride = stride,
        padding = padding,
    )
    return Batch(
        value = result,
        size = size,
        shape = listOf(
            shape[0],
            (shape[1] - windowSize + padding * 2) / stride + 1,
            windowSize,
        ),
    )
}

fun Batch<IOType.D3>.fold(stride: Int, padding: Int): Batch<IOType.D2> {
    val result = Backend.fold(
        x = value,
        xi = shape[0],
        xj = shape[1],
        xk = shape[2],
        b = size,
        stride = stride,
        padding = padding,
    )
    return Batch(
        value = result,
        size = size,
        shape = listOf(
            shape[0],
            shape[2] + (shape[1] - 1) * stride - padding * 2,
        )
    )
}
