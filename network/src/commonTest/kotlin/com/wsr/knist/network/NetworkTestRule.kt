package com.wsr.knist.network

import com.wsr.knist.Backend
import com.wsr.knist.BufferScope
import com.wsr.knist.base.KotlinBackend
import com.wsr.knist.core.IOScope
import kotlin.time.measureTime

fun networkTestRule(evaluate: () -> Unit) {
    Backend.set(backend = KotlinBackend)
    measureTime { evaluate() }
        .also { time -> println("$time") }
}

fun networkScopeTestRule(evaluate: IOScope.() -> Unit) {
    networkTestRule {
        BufferScope().use { bufferScope ->
            IOScope(bufferScope).evaluate()
        }
    }
}
