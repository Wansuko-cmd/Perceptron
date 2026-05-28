package com.wsr.gpu

import com.wsr.base.IBackend

actual fun loadGPUBackend(): IBackend? = runCatching<IBackend> {
    System.loadLibrary("gpu")
    GPUBackend()
}.getOrNull()
