package com.wsr.knist.gpu

object JRuntime {
    external fun allocate(): Long
    external fun release(ptr: Long)
    external fun flush(runtime: Long)
    external fun sync(runtime: Long)
}
