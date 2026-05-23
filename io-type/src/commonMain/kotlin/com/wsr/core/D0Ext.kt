package com.wsr.core

import com.wsr.base.data.DataBuffer

fun IOType.Companion.d0(value: Float) = IOType.d0(floatArrayOf(value))

fun IOType.Companion.d0(value: FloatArray) = IOType.D0(DataBuffer.create(value))

fun IOType.D0.get() = value[0]

fun IOType.D0.set(element: Float) {
    value[0] = element
}
