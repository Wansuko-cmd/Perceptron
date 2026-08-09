package com.wsr.knist.cpu

import com.wsr.knist.base.IBackend

expect fun loadCPUBackend(
    fallback: IBackend,
    maxReservedBytes: Long,
    maxPoolBytes: Long = maxReservedBytes,
): IBackend
