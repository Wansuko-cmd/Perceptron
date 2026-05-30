package com.wsr.knist.batch

import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import kotlin.jvm.JvmName

@JvmName("batchD0sGet")
operator fun Batch<IOType.D0>.get(i: Int): IOType.D0 {
    val index = i * step
    return IOType.d0(value[index])
}

operator fun Batch<IOType.D0>.set(i: Int, element: IOType.D0) {
    value[i] = element.value[0]
}
