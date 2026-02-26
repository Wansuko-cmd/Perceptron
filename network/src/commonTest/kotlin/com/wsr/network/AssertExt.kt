package com.wsr.network

import com.wsr.core.IOType
import kotlin.math.abs
import kotlin.test.fail

fun <T : IOType> assertContentEquals(expected: T, actual: T, absoluteTolerance: Float) {
    if (expected.shape != actual.shape) {
        fail("Expected <$expected> with absolute tolerance <$absoluteTolerance>, actual <$actual>.")
    }
    repeat(expected.size) {
        if (abs(expected.value[it] - actual.value[it]) > absoluteTolerance) {
            fail("Expected <$expected> with absolute tolerance <$absoluteTolerance>, actual <$actual>.")
        }
    }
}
