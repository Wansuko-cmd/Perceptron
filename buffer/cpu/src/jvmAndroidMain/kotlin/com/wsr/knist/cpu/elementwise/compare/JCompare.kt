package com.wsr.knist.cpu.elementwise.compare

object JCompare {
    external fun greaterThanD1ToD0(x: Long, y: Float, result: Long)
    external fun greaterThanD1ToD1(x: Long, y: Long, result: Long)

    external fun lessThanD1ToD0(x: Long, y: Float, result: Long)
    external fun lessThanD1ToD1(x: Long, y: Long, result: Long)

    external fun equalsD1ToD0(x: Long, y: Float, absoluteTolerance: Float, relativeTolerance: Float, result: Long)
    external fun equalsD1ToD1(x: Long, y: Long, absoluteTolerance: Float, relativeTolerance: Float, result: Long)
}
