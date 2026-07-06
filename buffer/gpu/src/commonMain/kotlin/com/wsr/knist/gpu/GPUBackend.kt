package com.wsr.knist.gpu

import com.wsr.knist.base.IBackend
import com.wsr.knist.base.KotlinBackend

val gpu: IBackend = loadGPUBackend(KotlinBackend)

expect fun loadGPUBackend(fallback: IBackend, enableProfiler: Boolean = false): IBackend
