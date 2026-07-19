package com.wsr.knist.cpu.elementwise.math


object JMath {
    external fun exp(x: Long, result: Long)
    external fun ln(x: Long, e: Float, result: Long)
    external fun sigmoid(x: Long, result: Long)
    external fun pow(x: Long, n: Int, result: Long)
    external fun sqrt(x: Long, e: Float, result: Long)
}
