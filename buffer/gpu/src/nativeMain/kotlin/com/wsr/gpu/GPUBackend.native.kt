package com.wsr.gpu

import com.wsr.base.IBackend
import com.wsr.base.KotlinBackend

actual fun loadGPUBackend(): IBackend? = KotlinBackend
