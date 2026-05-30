package com.wsr.knist.cpu.linalg

import java.nio.ByteBuffer

class JMatMul {
    external fun inner(x: ByteBuffer, y: ByteBuffer, b: Int, result: ByteBuffer)

    external fun matMulD1ToD2(x: ByteBuffer, y: ByteBuffer, transY: Boolean, n: Int, k: Int, result: ByteBuffer)

    external fun matMulD2ToD1(x: ByteBuffer, transX: Boolean, y: ByteBuffer, m: Int, k: Int, result: ByteBuffer)

    external fun matMulD2ToD2(
        x: ByteBuffer,
        transX: Boolean,
        y: ByteBuffer,
        transY: Boolean,
        m: Int,
        n: Int,
        k: Int,
        b: Int,
        result: ByteBuffer,
    )
}
