package com.wsr.knist.cpu

object JRuntime {
    external fun allocate(poolSize: Long): Long
    external fun release(ptr: Long)
}
