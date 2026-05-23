package com.wsr.batch

import com.wsr.Backend
import com.wsr.core.IOType
import kotlin.jvm.JvmName

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
