package com.wsr.knist.cpu.elementwise.compare

import java.nio.ByteBuffer

object JCompare {
    external fun greaterThanD1ToD0(x: ByteBuffer, y: Float, result: ByteBuffer)
    external fun greaterThanD1ToD1(x: ByteBuffer, y: ByteBuffer, result: ByteBuffer)

    external fun lessThanD1ToD0(x: ByteBuffer, y: Float, result: ByteBuffer)
    external fun lessThanD1ToD1(x: ByteBuffer, y: ByteBuffer, result: ByteBuffer)

    external fun equalsD1ToD0(
        x: ByteBuffer,
        y: Float,
        absoluteTolerance: Float,
        relativeTolerance: Float,
        result: ByteBuffer,
    )
    external fun equalsD1ToD1(
        x: ByteBuffer,
        y: ByteBuffer,
        absoluteTolerance: Float,
        relativeTolerance: Float,
        result: ByteBuffer,
    )
}
