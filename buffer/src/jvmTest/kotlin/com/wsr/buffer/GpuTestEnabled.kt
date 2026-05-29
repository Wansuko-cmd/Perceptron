package com.wsr.buffer

internal actual val isGpuEnabled: Boolean = System.getenv("SKIP_GPU_TESTS") != "true"
