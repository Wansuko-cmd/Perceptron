package com.wsr.knist.batch.shape

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName

fun IOType.D1.toBatch(): Batch<IOType.D0> = Batch(size = size, shape = listOf(1), value = value)

@JvmName("batchD1sToList")
fun Batch<IOType.D1>.toList(): List<IOType.D1> = List(size) { get(it) }
