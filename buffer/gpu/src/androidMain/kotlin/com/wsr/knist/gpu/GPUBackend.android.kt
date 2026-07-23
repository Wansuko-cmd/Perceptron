package com.wsr.knist.gpu

import com.wsr.knist.base.IBackend

actual fun loadGPUBackend(fallback: IBackend, enableProfiler: Boolean, maxReservedBytes: Long): IBackend =
    runCatching<IBackend> {
        System.loadLibrary("gpu")
        gpuMaxReservedBytes = maxReservedBytes
        GPUBackend(fallback, enableProfiler)
    }.getOrElse { fallback }
