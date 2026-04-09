package com.wsr.batch

import com.wsr.Backend
import com.wsr.base.data.DataBuffer
import com.wsr.core.IOType
import kotlin.jvm.JvmName

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

@JvmName("batchD1sToList")
fun Batch<IOType.D1>.toList(): List<IOType.D1> = List(size) { get(it) }

@JvmName("batchD2sToList")
fun Batch<IOType.D2>.toList(): List<IOType.D2> = List(size) { get(it) }

@JvmName("batchD3sToList")
fun Batch<IOType.D3>.toList(): List<IOType.D3> = List(size) { get(it) }
