package com.wsr.network

import com.wsr.batch.Batch
import com.wsr.core.IOType
import kotlin.math.abs
import kotlin.test.fail

fun <T : IOType> assertContentEquals(expected: T, actual: T, absoluteTolerance: Float = 1e-4f) {
    if (expected.shape != actual.shape) {
        fail("Expected <$expected> with absolute tolerance <$absoluteTolerance>, actual <$actual>.")
    }
    repeat(expected.size) {
        if (abs(expected.value[it] - actual.value[it]) > absoluteTolerance) {
            fail("Expected <$expected> with absolute tolerance <$absoluteTolerance>, actual <$actual>.")
        }
    }
}

fun <T : IOType> assertContentEquals(expected: Batch<T>, actual: Batch<T>, absoluteTolerance: Float = 1e-4f) {
    if (expected.shape != actual.shape) {
        fail("Expected <$expected> with absolute tolerance <$absoluteTolerance>, actual <$actual>.")
    }
    repeat(expected.size) {
        if (abs(expected.value[it] - actual.value[it]) > absoluteTolerance) {
            fail("Expected <$expected> with absolute tolerance <$absoluteTolerance>, actual <$actual>.")
        }
    }
}

fun <T : IOType> assertContentEquals(expected: List<T>, actual: List<T>, absoluteTolerance: Float = 1e-4f) {
    if (expected.size != actual.size) {
        fail("Expected <$expected> with absolute tolerance <$absoluteTolerance>, actual <$actual>.")
    }
    repeat(expected.size) {
        assertContentEquals(expected[it], actual[it])
    }
}
