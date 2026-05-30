package com.wsr.knist.gpu

import com.wsr.knist.base.IBackend

actual fun loadGPUBackend(): IBackend? = runCatching<IBackend> {
    System.loadLibrary("gpu")
    GPUBackend()
}.getOrNull()
