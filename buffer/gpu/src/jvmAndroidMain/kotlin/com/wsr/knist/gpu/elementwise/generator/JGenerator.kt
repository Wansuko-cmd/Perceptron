package com.wsr.knist.gpu.elementwise.generator

object JGenerator {
    external fun random(from: Float, until: Float, seed: Long, result: Long, runtime: Long)
}
