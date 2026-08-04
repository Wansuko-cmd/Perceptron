package com.wsr.knist.cpu

object JRuntime {
    external fun allocate(): Long
    external fun release(ptr: Long)
}
