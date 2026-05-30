package com.wsr.knist.cpu.elementwise.operation.div

import java.nio.ByteBuffer

class JDiv {
    external fun divD0ToD1(x: Float, y: ByteBuffer, result: ByteBuffer)

    external fun divD1ToD0(x: ByteBuffer, y: Float, result: ByteBuffer)
    external fun divD1ToD1(x: ByteBuffer, y: ByteBuffer, result: ByteBuffer)
    external fun divD1ToD2(x: ByteBuffer, y: ByteBuffer, yi: Int, yj: Int, axis: Int, result: ByteBuffer)
    external fun divD1ToD3(x: ByteBuffer, y: ByteBuffer, yi: Int, yj: Int, yk: Int, axis: Int, result: ByteBuffer)

    external fun divD2ToD1(x: ByteBuffer, xi: Int, xj: Int, y: ByteBuffer, axis: Int, result: ByteBuffer)
    external fun divD2ToD3(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        y: ByteBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        axis1: Int,
        axis2: Int,
        result: ByteBuffer,
    )

    external fun divD3ToD1(x: ByteBuffer, xi: Int, xj: Int, xk: Int, y: ByteBuffer, axis: Int, result: ByteBuffer)
    external fun divD3ToD2(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        y: ByteBuffer,
        yi: Int,
        yj: Int,
        axis1: Int,
        axis2: Int,
        result: ByteBuffer,
    )
    external fun divD3ToD4(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        y: ByteBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        yl: Int,
        axis1: Int,
        axis2: Int,
        axis3: Int,
        result: ByteBuffer,
    )

    external fun divD4ToD1(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: ByteBuffer,
        axis: Int,
        result: ByteBuffer,
    )
    external fun divD4ToD2(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: ByteBuffer,
        yi: Int,
        yj: Int,
        axis1: Int,
        axis2: Int,
        result: ByteBuffer,
    )
    external fun divD4ToD3(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: ByteBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        axis1: Int,
        axis2: Int,
        axis3: Int,
        result: ByteBuffer,
    )
}
