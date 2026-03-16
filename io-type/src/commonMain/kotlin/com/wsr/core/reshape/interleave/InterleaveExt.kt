package com.wsr.core.reshape.interleave

import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.core.IOType
import com.wsr.create

fun IOType.D1.interleave(other: IOType.D1): IOType.D1 {
    check(size == other.size)

    val result = DataBuffer.create(size + other.size)
    value.copyInto(dest = result, destIndices = result.indices step 2)
    other.value.copyInto(dest = result, destIndices = 1 until result.size step 2)

    return IOType.D1(result)
}

fun IOType.D2.interleave(other: IOType.D2, axis: Int): IOType.D2 {
    check(i == other.i && j == other.j)
    val result = DataBuffer.create(size + other.size)
    return when (axis) {
        0 -> {
            repeat(i) {
                val start = it * j
                val end = start + j
                value
                    .slice(start until end)
                    .run {
                        val destStart = it * j * 2
                        val destEnd = destStart + j
                        copyInto(result, destStart until destEnd)
                    }
                other.value
                    .slice(start until end)
                    .run {
                        val destStart = it * j * 2 + j
                        val destEnd = destStart + j
                        copyInto(result, destStart until destEnd)
                    }
            }
            IOType.D2(shape = listOf(i * 2, j), value = result)
        }
        else -> {
            repeat(i) {
                val start = it * j
                val end = start + j
                val destStart = it * j * 2
                val destEnd = destStart + j * 2
                value
                    .slice(start until end)
                    .copyInto(result, destStart until destEnd step 2)
                other.value
                    .slice(start until end)
                    .copyInto(result, destStart + 1 until destEnd step 2)
            }
            IOType.D2(shape = listOf(i, j * 2), value = result)
        }
    }
}
