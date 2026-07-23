package com.wsr.knist.cpu

import com.wsr.knist.base.IBackend
import com.wsr.knist.base.loadNativeLibrary

private const val LIB_PATH = "cpu"
private const val LIB_NAME = "cpu"

actual fun loadCPUBackend(fallback: IBackend, maxReservedBytes: Long): IBackend {
    val isSuccess = loadNativeLibrary(path = LIB_PATH, name = LIB_NAME)
    if (isSuccess) cpuMaxReservedBytes = maxReservedBytes
    return if (isSuccess) CPUJvmBackend(fallback) else fallback
}
