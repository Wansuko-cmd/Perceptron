package com.wsr.batch.shape

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.batch.Batch
import com.wsr.core.IOType

fun <T : IOType> List<T>.toBatch(): Batch<T> {
    val batchSize = size
    val shape = first().shape
    val step = shape.reduce { acc, i -> acc * i }
    val batchValue = DataBuffer.create(batchSize * step)
    forEachIndexed { index, item ->
        val start = index * step
        Backend.copyInto(item.value, batchValue, start until start + item.value.size)
    }
    return Batch(
        value = batchValue,
        size = batchSize,
        shape = shape,
    )
}
