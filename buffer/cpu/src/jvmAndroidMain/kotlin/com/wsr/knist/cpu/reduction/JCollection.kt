package com.wsr.knist.cpu.reduction

import java.nio.ByteBuffer

class JCollection {
    external fun averageD1(x: ByteBuffer): Float
    external fun averageD2(x: ByteBuffer, xi: Int, xj: Int, axis: Int, result: ByteBuffer)
    external fun averageD3(x: ByteBuffer, xi: Int, xj: Int, xk: Int, axis: Int, result: ByteBuffer)
    external fun averageD4(x: ByteBuffer, xi: Int, xj: Int, xk: Int, xl: Int, axis: Int, result: ByteBuffer)

    external fun maxD1(x: ByteBuffer): Float
    external fun maxD2(x: ByteBuffer, xi: Int, xj: Int, axis: Int, result: ByteBuffer)
    external fun maxD3(x: ByteBuffer, xi: Int, xj: Int, xk: Int, axis: Int, result: ByteBuffer)
    external fun maxD4(x: ByteBuffer, xi: Int, xj: Int, xk: Int, xl: Int, axis: Int, result: ByteBuffer)

    external fun minD1(x: ByteBuffer): Float
    external fun minD2(x: ByteBuffer, xi: Int, xj: Int, axis: Int, result: ByteBuffer)
    external fun minD3(x: ByteBuffer, xi: Int, xj: Int, xk: Int, axis: Int, result: ByteBuffer)
    external fun minD4(x: ByteBuffer, xi: Int, xj: Int, xk: Int, xl: Int, axis: Int, result: ByteBuffer)

    external fun sumD1(x: ByteBuffer): Float
    external fun sumD2(x: ByteBuffer, xi: Int, xj: Int, axis: Int, result: ByteBuffer)
    external fun sumD3(x: ByteBuffer, xi: Int, xj: Int, xk: Int, axis: Int, result: ByteBuffer)
    external fun sumD4(x: ByteBuffer, xi: Int, xj: Int, xk: Int, xl: Int, axis: Int, result: ByteBuffer)
}
