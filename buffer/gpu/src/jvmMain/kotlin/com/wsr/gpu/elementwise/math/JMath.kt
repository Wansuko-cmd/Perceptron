package com.wsr.gpu.elementwise.math

class JMath {
    external fun exp(x: Long, result: Long, runtime: Long)
    external fun ln(x: Long, e: Float, result: Long, runtime: Long)
    external fun sigmoid(x: Long, result: Long, runtime: Long)
    external fun pow(x: Long, n: Int, result: Long, runtime: Long)
    external fun sqrt(x: Long, e: Float, result: Long, runtime: Long)
}
