package com.wsr.knist.buffer

import com.wsr.knist.Backend
import com.wsr.knist.base.KotlinBackend
import com.wsr.knist.cpu.cpu
import com.wsr.knist.gpu.gpu
import kotlin.time.measureTime

private val targets = listOf(KotlinBackend, cpu, gpu)

fun bufferTestRule(evaluate: () -> Unit) {
    targets.forEach { target ->
        Backend.set(target)
        measureTime { evaluate() }
            .also { time -> println("$time($target)") }
    }
}
