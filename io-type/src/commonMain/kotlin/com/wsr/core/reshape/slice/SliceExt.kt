package com.wsr.core.reshape.slice

import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.core.IOType
import com.wsr.create
import kotlin.math.min

fun IOType.D1.slice(indices: IntProgression) = IOType.D1(value = value.slice(indices))

fun IOType.D2.slice(indices: IntProgression, axis: Int) = when (axis) {
    0 -> {
        val count = min(i, indices.size)
        val result = DataBuffer.create(count * j)
        repeat(j) {
            val start = indices.first * j + it
            val end = indices.last * j + it
            val step = indices.step * j
            value
                .slice(start..end step step)
                .copyInto(result, it until result.size step j)
        }
        IOType.D2(shape = listOf(count, j), value = result)
    }
    else -> {
        val count = min(j, indices.size)
        val result = DataBuffer.create(i * count)
        repeat(i) {
            val offset = it * j
            val start = offset + indices.first
            val end = offset + indices.last
            val destStart = it * count
            value
                .slice(start..end step indices.step)
                .copyInto(result, destStart until destStart + count)
        }
        IOType.D2(shape = listOf(i, count), value = result)
    }
}
