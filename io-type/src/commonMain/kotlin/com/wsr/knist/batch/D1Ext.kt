package com.wsr.knist.batch

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.core.D1
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import kotlin.jvm.JvmName

inline fun Batch.Companion.d1(batchSize: Int, size: Int, init: (Int) -> Float = { 0f }): Batch<IOType.D1> =
    Batch(batchSize) { IOType.d1(size, init) }

inline fun Batch.Companion.d1(batchSize: Int, shape: List<Int>, init: (Int) -> Float = { 0f }): Batch<IOType.D1> =
    d1(batchSize, shape[0], init)

fun Batch.Companion.d1(batchSize: Int, size: Int, value: FloatArray): Batch<IOType.D1> =
    Batch(value = DataBuffer.create(value), size = batchSize, shape = listOf(size))

internal fun Batch.Companion.d1(batchSize: Int, size: Int, value: DataBuffer): Batch<IOType.D1> =
    Batch(value = value, size = batchSize, shape = listOf(size))

internal fun Batch.Companion.d1(batchSize: Int, shape: List<Int>, value: DataBuffer): Batch<IOType.D1> =
    Batch(value = value, size = batchSize, shape = shape)

val Batch<IOType.D1>.i get() = shape[0]

@JvmName("batchD1sGet")
operator fun Batch<IOType.D1>.get(i: Int): IOType.D1 {
    val index = i * step
    val result = Backend.slice(x = value, indices = index until index + step)
    return IOType.D1(result)
}

operator fun Batch<IOType.D1>.set(i: Int, element: IOType.D1) {
    val start = i * step
    Backend.copyInto(element.value, value, start until start + element.value.size)
}
