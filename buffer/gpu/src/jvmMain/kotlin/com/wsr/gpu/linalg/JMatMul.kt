package com.wsr.gpu.linalg

class JMatMul {
    external fun matMul(
        x: Long,
        transX: Boolean,
        y: Long,
        transY: Boolean,
        m: Int,
        n: Int,
        k: Int,
        b: Int,
        result: Long,
        runtime: Long,
    )
}
