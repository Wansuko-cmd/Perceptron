package com.wsr.batch.shape

import com.wsr.batch.Batch
import com.wsr.core.IOType

fun IOType.D4.toBatch(): Batch<IOType.D3> =
    Batch(value = value, size = shape[0], shape = listOf(shape[1], shape[2], shape[3]))
