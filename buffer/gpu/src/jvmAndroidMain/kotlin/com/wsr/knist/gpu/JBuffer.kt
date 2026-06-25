package com.wsr.knist.gpu

object JBuffer {
    external fun allocate(size: Int, runtime: Long): Long
    external fun init(value: FloatArray, runtime: Long): Long
    external fun release(ptr: Long, runtime: Long)

    external fun readAll(ptr: Long, runtime: Long): FloatArray
    external fun write(ptr: Long, index: Int, value: Float, runtime: Long)
}
