package com.wsr.knist.cpu

import com.wsr.knist.base.IBackend

actual fun loadCPUBackend(fallback: IBackend): IBackend = runCatching<IBackend> {
    System.loadLibrary("cpu")
    CPUJvmBackend(fallback)
}.getOrElse { fallback }
