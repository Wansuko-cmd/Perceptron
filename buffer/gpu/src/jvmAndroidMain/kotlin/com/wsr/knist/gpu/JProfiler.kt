package com.wsr.knist.gpu

object JProfiler {
    external fun takeCPU(runtime: Long): Long
    external fun takeGPU(runtime: Long): LongArray
}

data class GPUProfilerResult(val busyNs: Long, val spanNs: Long)
