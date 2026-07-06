package com.wsr.knist.gpu

import com.wsr.knist.base.IBackend

actual fun loadGPUBackend(fallback: IBackend, enableProfiler: Boolean): IBackend = runCatching<IBackend> {
    System.loadLibrary("gpu")
    GPUBackend(fallback, enableProfiler)
}.getOrElse { fallback }
