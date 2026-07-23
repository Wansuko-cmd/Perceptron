package com.wsr.knist.gpu

import com.wsr.knist.base.IBackend

expect fun loadGPUBackend(
    fallback: IBackend,
    enableProfiler: Boolean = false,
    maxReservedBytes: Long = 4_000_000_000L,
): IBackend
