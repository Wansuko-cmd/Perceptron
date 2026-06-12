package com.wsr.knist.batch

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.D2
import kotlin.jvm.JvmName

val Batch<IOType.D2>.i get() = shape[0]

val Batch<IOType.D2>.j get() = shape[1]

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
