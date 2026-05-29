package com.wsr.buffer

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import platform.posix.getenv

@OptIn(ExperimentalForeignApi::class)
internal actual val isGpuEnabled: Boolean = getenv("SKIP_GPU_TESTS")?.toKString() != "true"
