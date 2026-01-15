package com.wsr.buffer

import com.wsr.Backend
import com.wsr.base.KotlinBackend
import com.wsr.cpu.cpu
import kotlin.time.measureTime

private val targets = listOf(KotlinBackend, cpu)

fun bufferTestRule(evaluate: () -> Unit) {
    targets.forEach { target ->
        Backend.set(target)
        measureTime { evaluate() }
            .also { time -> println("$time($target)") }
    }
}
