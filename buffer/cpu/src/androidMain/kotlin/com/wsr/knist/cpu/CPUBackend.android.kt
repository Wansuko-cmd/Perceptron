package com.wsr.knist.cpu

import com.wsr.knist.base.IBackend

actual fun loadCPUBackend(fallback: IBackend, maxReservedBytes: Long): IBackend = runCatching<IBackend> {
    System.loadLibrary("cpu")
    cpuMaxReservedBytes = maxReservedBytes
    CPUJvmBackend(fallback)
}.getOrElse { fallback }
