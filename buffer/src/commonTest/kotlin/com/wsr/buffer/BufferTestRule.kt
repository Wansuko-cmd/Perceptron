package com.wsr.buffer

import com.wsr.Backend
import com.wsr.base.KotlinBackend
import com.wsr.cpu.cpu
import com.wsr.gpu.gpu
import kotlin.time.measureTime

private val targets = listOf(KotlinBackend, cpu, gpu)

fun bufferTestRule(evaluate: () -> Unit) {
    targets.forEach { target ->
        Backend.set(target)
        measureTime { evaluate() }
            .also { time -> println("$time($target)") }
    }
}
