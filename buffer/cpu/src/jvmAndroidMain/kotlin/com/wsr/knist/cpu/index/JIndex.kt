package com.wsr.knist.cpu.index

import java.nio.ByteBuffer

class JIndex {
    external fun gather(x: ByteBuffer, y: ByteBuffer, i: Int, j: Int, k: Int, result: ByteBuffer)
    external fun scatterAdd(x: ByteBuffer, y: ByteBuffer, i: Int, j: Int, k: Int, b: Int, result: ByteBuffer)
}
