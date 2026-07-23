package com.wsr.knist.cpu

import com.wsr.knist.base.IBackend

internal expect fun defaultMaxReservedBytes(): Long

expect fun loadCPUBackend(fallback: IBackend, maxReservedBytes: Long = defaultMaxReservedBytes()): IBackend
