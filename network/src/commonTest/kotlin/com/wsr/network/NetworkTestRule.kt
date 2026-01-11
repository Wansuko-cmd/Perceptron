package com.wsr.network

import com.wsr.Backend
import com.wsr.base.KotlinBackend
import kotlin.time.measureTime
import org.junit.rules.TestRule
import org.junit.runner.Description
import org.junit.runners.model.Statement

class NetworkTestRule : TestRule {
    override fun apply(base: Statement?, description: Description?): Statement = object : Statement() {
        override fun evaluate() {
            Backend.set(backend = KotlinBackend)
            measureTime { base?.evaluate() }
                .also { time -> println("${description?.className}::${description?.methodName}: $time") }
        }
    }
}
