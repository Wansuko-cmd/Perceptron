package com.wsr.knist.cpu.shape

import java.nio.ByteBuffer

object JShape {
    external fun transposeD2(x: ByteBuffer, xi: Int, xj: Int, result: ByteBuffer)

    external fun transposeD3(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        axisI: Int,
        axisJ: Int,
        axisK: Int,
        result: ByteBuffer,
    )

    external fun transposeD4(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        axisI: Int,
        axisJ: Int,
        axisK: Int,
        axisL: Int,
        result: ByteBuffer,
    )

    external fun transposeD5(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        xm: Int,
        axisI: Int,
        axisJ: Int,
        axisK: Int,
        axisL: Int,
        axisM: Int,
        result: ByteBuffer,
    )

    external fun sliceD1(x: ByteBuffer, start: Int, end: Int, step: Int, result: ByteBuffer)

    external fun sliceD2(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        axis: Int,
        start: Int,
        end: Int,
        step: Int,
        result: ByteBuffer,
    )

    external fun sliceD3(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        axis: Int,
        start: Int,
        end: Int,
        step: Int,
        result: ByteBuffer,
    )

    external fun copyIntoD1(x: ByteBuffer, result: ByteBuffer, start: Int, end: Int, step: Int)

    external fun copyIntoD2(
        x: ByteBuffer,
        result: ByteBuffer,
        ri: Int,
        rj: Int,
        axis: Int,
        start: Int,
        end: Int,
        step: Int,
    )

    external fun copyIntoD3(
        x: ByteBuffer,
        result: ByteBuffer,
        ri: Int,
        rj: Int,
        rk: Int,
        axis: Int,
        start: Int,
        end: Int,
        step: Int,
    )

    external fun unfoldD1(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        b: Int,
        window: Int,
        stride: Int,
        dilation: Int,
        padding: Int,
        result: ByteBuffer,
    )

    external fun unfoldD2(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        b: Int,
        window: Int,
        stride: Int,
        dilation: Int,
        padding: Int,
        result: ByteBuffer,
    )

    external fun foldD1(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        b: Int,
        stride: Int,
        dilation: Int,
        padding: Int,
        result: ByteBuffer,
    )

    external fun foldD2(
        x: ByteBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        b: Int,
        stride: Int,
        dilation: Int,
        padding: Int,
        result: ByteBuffer,
    )

    external fun flipD3(x: ByteBuffer, xi: Int, xj: Int, xk: Int, axis: Int, result: ByteBuffer)
}
