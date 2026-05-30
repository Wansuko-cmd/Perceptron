@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.initializer

import com.wsr.knist.core.get
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class HeTest {
    @Test
    fun `d1=Heアルゴリズムで初期化`() = networkTestRule {
        val target = He(seed = 0)

        val actual = target.d1(input = listOf(3), output = listOf(3), size = 3)

        assertEquals(expected = 0.1403f, actual = actual[0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9867f, actual = actual[1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.4500f, actual = actual[2], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `d2=Heアルゴリズムで初期化`() = networkTestRule {
        val target = He(seed = 0)

        val actual = target.d2(input = listOf(2, 2), output = listOf(2, 2), x = 2, y = 2)

        assertEquals(expected = 0.1215f, actual = actual[0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.8545f, actual = actual[0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3897f, actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -1.2051f, actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `d3=Heアルゴリズムで初期化`() = networkTestRule {
        val target = He(seed = 0)

        val actual = target.d3(input = listOf(2, 2, 2), output = listOf(2, 2, 2), x = 2, y = 2, z = 2)

        assertEquals(expected = 0.0859f, actual = actual[0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.6042f, actual = actual[0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2755f, actual = actual[0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.8521f, actual = actual[0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.3031f, actual = actual[1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.8485f, actual = actual[1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.0598f, actual = actual[1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3319f, actual = actual[1][1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `d4=Heアルゴリズムで初期化`() = networkTestRule {
        val target = He(seed = 0)

        val actual = target.d4(input = listOf(2, 2, 2, 2), output = listOf(2, 2, 2, 2), i = 2, j = 2, k = 2, l = 2)

        assertEquals(expected = 0.0607f, actual = actual[0][0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.4272f, actual = actual[0][0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.1948f, actual = actual[0][0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.6025f, actual = actual[0][0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.2143f, actual = actual[0][1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.5999f, actual = actual[0][1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.0423f, actual = actual[0][1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2347f, actual = actual[0][1][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.5491f, actual = actual[1][0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.6104f, actual = actual[1][0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.4854f, actual = actual[1][0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.0592f, actual = actual[1][0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.0221f, actual = actual[1][1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.2305f, actual = actual[1][1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.3889f, actual = actual[1][1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.1772f, actual = actual[1][1][1][1], absoluteTolerance = 1e-4f)
    }
}
