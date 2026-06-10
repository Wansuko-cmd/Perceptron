package com.wsr.knist.batch.shape

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

fun IOType.D1.toBatch(): Batch<IOType.D0> = Batch(size = size, shape = listOf(1), value = value)

@JvmName("batchD0sToList")
fun Batch<IOType.D0>.toList(): List<IOType.D0> = List(size) { get(it) }
