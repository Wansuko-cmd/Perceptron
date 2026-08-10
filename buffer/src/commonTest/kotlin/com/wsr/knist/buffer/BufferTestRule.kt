package com.wsr.knist.buffer

import com.wsr.knist.Backend
import com.wsr.knist.base.KotlinBackend
import com.wsr.knist.cpu.loadCPUBackend
import com.wsr.knist.gpu.loadGPUBackend
import kotlin.time.measureTime

private val targets = listOf(
    KotlinBackend,
    loadCPUBackend(fallback = KotlinBackend),
    loadGPUBackend(fallback = KotlinBackend),
)

fun bufferTestRule(evaluate: () -> Unit) {
    targets.forEach { target ->
        Backend.set(target)
        measureTime { evaluate() }
            .also { time -> println("$time($target)") }
    }
}
