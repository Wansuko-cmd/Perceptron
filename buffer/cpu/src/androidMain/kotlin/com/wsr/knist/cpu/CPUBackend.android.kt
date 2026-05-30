package com.wsr.knist.cpu

import com.wsr.knist.base.IBackend

actual fun loadCPUBackend(): IBackend? = runCatching<IBackend> {
    System.loadLibrary("cpu")
    CPUJvmBackend()
}.getOrNull()
