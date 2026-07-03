package com.wsr.knist.cpu.elementwise.generator

import java.nio.ByteBuffer

object JGenerator {
    external fun random(from: Float, until: Float, seed: Long, result: ByteBuffer)
}
