package com.wsr.gpu

import com.wsr.base.IBackend
import com.wsr.base.KotlinBackend
import com.wsr.base.data.DataBuffer
import com.wsr.base.loadNativeLibrary

private const val LIB_PATH = "gpu"
private const val LIB_NAME = "gpu"

actual fun loadGPUBackend(): IBackend? = GPUBackend()

class GPUBackend : IBackend by KotlinBackend
