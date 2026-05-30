package com.wsr.knist.cpu

import com.wsr.knist.base.IBackend
import com.wsr.knist.base.KotlinBackend

val cpu: IBackend = loadCPUBackend() ?: KotlinBackend

expect fun loadCPUBackend(): IBackend?
