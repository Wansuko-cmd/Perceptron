package com.wsr.cpu

import com.wsr.base.IBackend
import com.wsr.base.KotlinBackend

actual fun loadCPUBackend(): IBackend? = KotlinBackend
