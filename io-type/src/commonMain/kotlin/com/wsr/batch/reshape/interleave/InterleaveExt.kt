package com.wsr.batch.reshape.interleave

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.batch.Batch
import com.wsr.core.IOType
import com.wsr.create

fun Batch<IOType.D1>.interleave(other: Batch<IOType.D1>): Batch<IOType.D1> {
    check(size == other.size && shape[0] == other.shape[0])
    val i = shape[0]
    val result = DataBuffer.create(value.size + other.value.size)
    repeat(size) {
        val start = it * i
        val end = start + i
        val destStart = it * i * 2
        val destEnd = destStart + i * 2
        Backend.slice(x = value, indices = start until end)
            .copyInto(result, destStart until destEnd step 2)
        Backend.slice(x = other.value, indices = start until end)
            .copyInto(result, destStart + 1 until destEnd step 2)
    }
    return Batch(size = size, shape = listOf(i * 2), value = result)
}

fun Batch<IOType.D2>.interleave(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
    check(size == other.size && shape[0] == other.shape[0] && shape[1] == other.shape[1])
    val (i, j) = shape
    val result = DataBuffer.create(value.size + other.value.size)
    return when (axis) {
        0 -> {
            repeat(size) { b ->
                val offset = b * i * j
                val destOffset = b * (i * 2) * j
                repeat(i) {
                    val start = offset + it * j
                    val end = start + j
                    Backend.slice(x = value, indices = start until end)
                        .run {
                            val destStart = destOffset + it * j * 2
                            val destEnd = destStart + j
                            copyInto(result, destStart until destEnd)
                        }
                    Backend.slice(x = other.value, indices = start until end)
                        .run {
                            val destStart = destOffset + it * j * 2 + j
                            val destEnd = destStart + j
                            copyInto(result, destStart until destEnd)
                        }
                }
            }
            Batch(size = size, shape = listOf(i * 2, j), value = result)
        }
        else -> {
            repeat(size * i) {
                val start = it * j
                val end = start + j
                val destStart = it * j * 2
                val destEnd = destStart + j * 2
                Backend.slice(x = value, indices = start until end)
                    .copyInto(result, destStart until destEnd step 2)
                Backend.slice(x = other.value, indices = start until end)
                    .copyInto(result, destStart + 1 until destEnd step 2)
            }
            Batch(size = size, shape = listOf(i, j * 2), value = result)
        }
    }
}
