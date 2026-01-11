@file:Suppress("NonAsciiCharacters")

package com.wsr.network.initializer

import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class XavierTest {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    @Test
    fun `d1=Xavierアルゴリズムで初期化`() {
        val target = Xavier(seed = 0)

        val actual = target.d1(input = listOf(3), output = listOf(3), size = 3)

        assertEquals(expected = 0.0992f, actual = actual[0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.6977f, actual = actual[1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3182f, actual = actual[2], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `d2=Xavierアルゴリズムで初期化`() {
        val target = Xavier(seed = 0)

        val actual = target.d2(input = listOf(2, 2), output = listOf(2, 2), x = 2, y = 2)

        assertEquals(expected = 0.0859f, actual = actual[0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.6042f, actual = actual[0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2755f, actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.8521f, actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `d3=Xavierアルゴリズムで初期化`() {
        val target = Xavier(seed = 0)

        val actual = target.d3(input = listOf(2, 2, 2), output = listOf(2, 2, 2), x = 2, y = 2, z = 2)

        assertEquals(expected = 0.0607f, actual = actual[0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.4272f, actual = actual[0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.1948f, actual = actual[0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.6025f, actual = actual[0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.2143f, actual = actual[1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.5999f, actual = actual[1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.0423f, actual = actual[1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2347f, actual = actual[1][1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `d4=Xavierアルゴリズムで初期化`() {
        val target = Xavier(seed = 0)

        val actual = target.d4(input = listOf(2, 2, 2, 2), output = listOf(2, 2, 2, 2), i = 2, j = 2, k = 2, l = 2)

        assertEquals(expected = 0.0429f, actual = actual[0][0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3021f, actual = actual[0][0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.1377f, actual = actual[0][0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.4260f, actual = actual[0][0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.1515f, actual = actual[0][1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.4242f, actual = actual[0][1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.0299f, actual = actual[0][1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.1659f, actual = actual[0][1][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3883f, actual = actual[1][0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.4316f, actual = actual[1][0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3432f, actual = actual[1][0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.0419f, actual = actual[1][0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.0156f, actual = actual[1][1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.1630f, actual = actual[1][1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.2750f, actual = actual[1][1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.1253f, actual = actual[1][1][1][1], absoluteTolerance = 1e-4f)
    }
}
