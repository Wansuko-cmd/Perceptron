package com.wsr.batch.shape

import com.wsr.batch.Batch
import com.wsr.core.IOType

fun IOType.D1.toBatch(): Batch<IOType.D0> = Batch(size = size, shape = listOf(1), value = value)
