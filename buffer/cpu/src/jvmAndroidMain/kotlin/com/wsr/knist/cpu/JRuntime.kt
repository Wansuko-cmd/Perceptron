package com.wsr.knist.cpu

object JRuntime {
    external fun allocate(poolSize: Int): Long
    external fun release(ptr: Long)
}
