package com.wsr.knist.batch

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.D3
import kotlin.jvm.JvmName

val Batch<IOType.D3>.i get() = shape[0]

val Batch<IOType.D3>.j get() = shape[1]

val Batch<IOType.D3>.k get() = shape[2]

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
