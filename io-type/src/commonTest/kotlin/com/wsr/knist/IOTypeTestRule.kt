package com.wsr.knist

import com.wsr.knist.base.KotlinBackend
import kotlin.time.measureTime

fun ioTypeTestRule(evaluate: () -> Unit) {
    Backend.set(backend = KotlinBackend)
    measureTime { evaluate() }
        .also { time -> println("$time") }
}
