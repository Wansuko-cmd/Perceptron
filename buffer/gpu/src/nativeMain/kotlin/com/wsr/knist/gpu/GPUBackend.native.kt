package com.wsr.knist.gpu

import com.wsr.knist.base.IBackend
import com.wsr.knist.base.KotlinBackend

actual fun loadGPUBackend(): IBackend? = KotlinBackend
