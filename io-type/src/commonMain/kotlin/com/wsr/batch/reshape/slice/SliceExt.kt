package com.wsr.batch.reshape.slice

import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.batch.Batch
import com.wsr.core.IOType
import com.wsr.create
import kotlin.math.min

fun Batch<IOType.D1>.slice(indices: IntProgression): Batch<IOType.D1> {
    val count = min(shape[0], indices.size)
    val result = DataBuffer.create(size * count)
    repeat(size) {
        val offset = it * shape[0]
        val start = offset + indices.first
        val end = offset + indices.last
        value
            .slice(start..end step indices.step)
            .copyInto(result, it * count until it * count + count)
    }
    return Batch(shape = listOf(count), size = size, value = result)
}

fun Batch<IOType.D2>.slice(indices: IntProgression, axis: Int): Batch<IOType.D2> = when (axis) {
    0 -> {
        val count = min(shape[0], indices.size)
        val result = DataBuffer.create(size * count * shape[1])

        val i = shape[0]
        val j = shape[1]
        repeat(size) { b ->
            val srcOffset = b * i * j
            val destOffset = b * count * j

            repeat(j) {
                val start = srcOffset + indices.first * j + it
                val end = srcOffset + indices.last * j + it
                val step = indices.step * j
                value
                    .slice(start..end step step)
                    .copyInto(result, destOffset + it until destOffset + count * j step j)
            }
        }
        Batch(size = size, shape = listOf(count, shape[1]), value = result)
    }
    else -> {
        val count = min(shape[1], indices.size)
        val result = DataBuffer.create(size * shape[0] * count)
        repeat(size * shape[0]) {
            val offset = it * shape[1]
            val start = offset + indices.first
            val end = offset + indices.last
            val destStart = it * count
            value
                .slice(start..end step indices.step)
                .copyInto(result, destStart until destStart + count)
        }
        Batch(shape = listOf(shape[0], count), size = size, value = result)
    }
}
