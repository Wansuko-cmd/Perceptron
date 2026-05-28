package com.wsr.gpu

class JRuntime {
    external fun allocate(): Long
    external fun release(ptr: Long)
}
