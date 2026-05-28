package com.wsr.cpu

import com.wsr.base.IBackend

actual fun loadCPUBackend(): IBackend? = runCatching<IBackend> {
    System.loadLibrary("cpu")
    CPUJvmBackend()
}.getOrNull()
