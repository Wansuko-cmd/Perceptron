package com.wsr.buffer

import com.wsr.base.data.DataBuffer
import kotlin.math.abs
import kotlin.test.fail

fun assertEquals(expected: DataBuffer, actual: DataBuffer, absoluteTolerance: Float) {
    if (expected.size != actual.size) {
        fail("Expected <$expected> with absolute tolerance <$absoluteTolerance>, actual <$actual>.")
    }
    repeat(expected.size) {
        if (abs(expected[it] - actual[it]) > absoluteTolerance) {
            fail("Expected <$expected> with absolute tolerance <$absoluteTolerance>, actual <$actual>.")
        }
    }
}
