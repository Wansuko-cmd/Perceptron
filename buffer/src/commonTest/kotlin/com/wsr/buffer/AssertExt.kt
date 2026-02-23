package com.wsr.buffer

import com.wsr.base.data.DataBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.fail

fun assertEquals(expected: DataBuffer, actual: DataBuffer, relativeTolerance: Float) {
    if (expected.size != actual.size) {
        fail("Expected <$expected> with absolute tolerance <$relativeTolerance>, actual <$actual>.")
    }
    repeat(expected.size) {
        val e = expected[it]
        val a = actual[it]
        val tolerance = relativeTolerance * max(abs(e), abs(a))
        if (abs(e - a) > tolerance) {
            fail("Expected <$expected> with absolute tolerance <$relativeTolerance>, actual <$actual>.")
        }
    }
}
