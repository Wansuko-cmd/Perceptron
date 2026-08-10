package com.wsr.knist.cpu

internal object JBuffer {
    external fun allocate(size: Int, runtime: Long): Long
    external fun release(ptr: Long, runtime: Long)

    external fun get(ptr: Long, index: Int): Float
    external fun set(ptr: Long, index: Int, value: Float)

    external fun readAll(ptr: Long): FloatArray
    external fun writeAll(ptr: Long, value: FloatArray)
}
