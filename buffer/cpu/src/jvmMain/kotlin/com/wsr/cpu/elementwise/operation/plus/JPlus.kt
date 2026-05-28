package com.wsr.cpu.elementwise.operation.plus

import java.nio.ByteBuffer

class JPlus {
    external fun plusD0ToD1(x: Float, y: ByteBuffer, result: ByteBuffer)

    external fun plusD1ToD0(x: ByteBuffer, y: Float, result: ByteBuffer)
    external fun plusD1ToD1(x: ByteBuffer, y: ByteBuffer, result: ByteBuffer)
    external fun plusD1ToD2(x: ByteBuffer, y: ByteBuffer, yi: Int, yj: Int, axis: Int, result: ByteBuffer)
    external fun plusD1ToD3(x: ByteBuffer, y: ByteBuffer, yi: Int, yj: Int, yk: Int, axis: Int, result: ByteBuffer)

    external fun plusD2ToD1(x: ByteBuffer, xi: Int, xj: Int, y: ByteBuffer, axis: Int, result: ByteBuffer)
    external fun plusD2ToD3(
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

    external fun plusD3ToD1(x: ByteBuffer, xi: Int, xj: Int, xk: Int, y: ByteBuffer, axis: Int, result: ByteBuffer)
    external fun plusD3ToD2(
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
    external fun plusD3ToD4(
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

    external fun plusD4ToD1(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: ByteBuffer,
        axis: Int,
        result: ByteBuffer,
    )
    external fun plusD4ToD2(
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
    external fun plusD4ToD3(
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
