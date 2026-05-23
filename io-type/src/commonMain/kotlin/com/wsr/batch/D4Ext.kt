package com.wsr.batch

import com.wsr.Backend
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD4sGet")
operator fun Batch<IOType.D4>.get(i: Int): IOType.D4 {
    val index = i * step
    val result = Backend.slice(x = value, indices = index until index + step)
    return IOType.D4(shape = shape, value = result)
}

operator fun Batch<IOType.D4>.set(i: Int, element: IOType.D4) {
    val start = i * step
    Backend.copyInto(element.value, value, start until start + element.value.size)
}
