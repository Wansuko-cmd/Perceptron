package com.wsr.knist.gpu

class JRuntime {
    external fun allocate(): Long
    external fun release(ptr: Long)
}
