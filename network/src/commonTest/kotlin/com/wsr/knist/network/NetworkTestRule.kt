package com.wsr.knist.network

import com.wsr.knist.Backend
import com.wsr.knist.base.KotlinBackend
import kotlin.time.measureTime

fun networkTestRule(evaluate: () -> Unit) {
    Backend.set(backend = KotlinBackend)
    measureTime { evaluate() }
        .also { time -> println("$time") }
}
