package com.wsr.batch

import com.wsr.Backend
import com.wsr.core.IOType
import kotlin.jvm.JvmName

@JvmName("batchD3sGet")
operator fun Batch<IOType.D3>.get(i: Int): IOType.D3 {
    val index = i * step
    val result = Backend.slice(x = value, indices = index until index + step)
    return IOType.D3(shape = shape, value = result)
}

operator fun Batch<IOType.D3>.set(i: Int, element: IOType.D3) {
    val start = i * step
    Backend.copyInto(element.value, value, start until start + element.value.size)
}
