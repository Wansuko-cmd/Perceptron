package com.wsr.network

import com.wsr.Backend
import com.wsr.base.KotlinBackend
import kotlin.time.measureTime

fun networkTestRule(evaluate: () -> Unit) {
    Backend.set(backend = KotlinBackend)
    measureTime { evaluate() }
        .also { time -> println("$time") }
}
