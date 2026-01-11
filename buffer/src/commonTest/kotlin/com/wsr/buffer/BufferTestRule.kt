package com.wsr.buffer

import com.wsr.Backend
import com.wsr.base.KotlinBackend
import com.wsr.cpu.cpu
import com.wsr.gpu.gpu
import kotlin.time.measureTime
import org.junit.rules.TestRule
import org.junit.runner.Description
import org.junit.runners.model.Statement

private val targets = listOf(KotlinBackend, cpu)

class BufferTestRule : TestRule {
    override fun apply(base: Statement?, description: Description?): Statement = object : Statement() {
        override fun evaluate() {
            targets.forEach { target ->
                Backend.set(target)
                measureTime { base?.evaluate() }
                    .also { time ->
                        println(
                            "${description?.className}::${description?.methodName}(${target::class.simpleName}): $time",
                        )
                    }
            }
        }
    }
}
