package com.wsr.knist.gpu

import com.wsr.knist.base.IBackend

actual fun loadGPUBackend(fallback: IBackend): IBackend = runCatching<IBackend> {
    System.loadLibrary("gpu")
    GPUBackend(fallback)
}.getOrElse { fallback }
