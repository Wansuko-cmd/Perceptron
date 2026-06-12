package com.wsr.knist.batch

import com.wsr.knist.Backend
import com.wsr.knist.core.IOType
import com.wsr.knist.core.D4
import kotlin.jvm.JvmName

val Batch<IOType.D4>.i get() = shape[0]

val Batch<IOType.D4>.j get() = shape[1]

val Batch<IOType.D4>.k get() = shape[2]

val Batch<IOType.D4>.l get() = shape[3]

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
