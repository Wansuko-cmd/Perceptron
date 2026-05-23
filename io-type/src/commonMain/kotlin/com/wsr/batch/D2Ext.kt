package com.wsr.batch

import com.wsr.Backend
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD2sGet")
operator fun Batch<IOType.D2>.get(i: Int): IOType.D2 {
    val index = i * step
    val result = Backend.slice(x = value, indices = index until index + step)
    return IOType.D2(shape = shape, value = result)
}

operator fun Batch<IOType.D2>.set(i: Int, element: IOType.D2) {
    val start = i * step
    Backend.copyInto(element.value, value, start until start + element.value.size)
}
