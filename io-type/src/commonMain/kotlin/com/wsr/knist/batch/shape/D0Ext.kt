package com.wsr.knist.batch.shape

import com.wsr.knist.Backend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

fun IOType.D1.toBatch(): Batch<IOType.D0> = Batch(size = size, shape = listOf(1), value = value)

@JvmName("batchD0sToList")
fun Batch<IOType.D0>.toList(): List<IOType.D0> = List(size) { get(it) }

fun Batch<IOType.D0>.broadcastToD1(size: Int): Batch<IOType.D1> {
    val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size, j = 1, k = 1)
    return Batch(size = this.size, shape = listOf(size), value = result)
}

fun Batch<IOType.D0>.broadcastToD2(shape: List<Int>): Batch<IOType.D2> {
    val size = shape.reduce { acc, i -> acc * i }
    val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size, j = 1, k = 1)
    return Batch(size = this.size, shape = shape, value = result)
}

fun Batch<IOType.D0>.broadcastToD3(shape: List<Int>): Batch<IOType.D3> {
    val size = shape.reduce { acc, i -> acc * i }
    val result = Backend.gather(x = DataBuffer.create(size), y = value, i = this.size, j = 1, k = 1)
    return Batch(size = this.size, shape = shape, value = result)
}
