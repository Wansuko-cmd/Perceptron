package com.wsr.buffer

import com.wsr.Backend
import com.wsr.base.KotlinBackend
import com.wsr.cpu.cpu
import com.wsr.gpu.gpu
import kotlin.time.measureTime

private val targets = buildList {
    add(KotlinBackend)
    add(cpu)
    if (isGpuEnabled) add(gpu)
}

internal expect val isGpuEnabled: Boolean

fun bufferTestRule(evaluate: () -> Unit) {
    targets.forEach { target ->
        Backend.set(target)
        measureTime { evaluate() }
            .also { time -> println("$time($target)") }
    }
}
