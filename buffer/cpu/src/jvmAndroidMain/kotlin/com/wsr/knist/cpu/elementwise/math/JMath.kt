package com.wsr.knist.cpu.elementwise.math

import java.nio.ByteBuffer

class JMath {
    external fun exp(x: ByteBuffer, result: ByteBuffer)
    external fun ln(x: ByteBuffer, e: Float, result: ByteBuffer)
    external fun sigmoid(x: ByteBuffer, result: ByteBuffer)
    external fun pow(x: ByteBuffer, n: Int, result: ByteBuffer)
    external fun sqrt(x: ByteBuffer, e: Float, result: ByteBuffer)
}
