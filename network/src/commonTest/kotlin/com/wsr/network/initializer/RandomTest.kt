@file:Suppress("NonAsciiCharacters")

package com.wsr.network.initializer

import com.wsr.core.get
import com.wsr.initializer.Random
import com.wsr.network.NetworkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class RandomTest {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    @Test
    fun `d1=ランダムに初期化`() {
        val target = Random(seed = 0)

        val actual = target.d1(input = listOf(3), output = listOf(3), size = 3)

        println(actual)

        assertEquals(expected = 0.0992f, actual = actual[0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.6977f, actual = actual[1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3182f, actual = actual[2], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `d2=ランダムに初期化`() {
        val target = Random(seed = 0)

        val actual = target.d2(input = listOf(2, 2), output = listOf(2, 2), x = 2, y = 2)

        assertEquals(expected = 0.0992f, actual = actual[0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.6977f, actual = actual[0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3182f, actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.9840f, actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `d3=ランダムに初期化`() {
        val target = Random(seed = 0)

        val actual = target.d3(input = listOf(2, 2, 2), output = listOf(2, 2, 2), x = 2, y = 2, z = 2)

        assertEquals(expected = 0.0992f, actual = actual[0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.6977f, actual = actual[0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3182f, actual = actual[0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.9840f, actual = actual[0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.3500f, actual = actual[1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.9797f, actual = actual[1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.0691f, actual = actual[1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3833f, actual = actual[1][1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `d4=ランダムに初期化`() {
        val target = Random(seed = 0)

        val actual = target.d4(input = listOf(2, 2, 2, 2), output = listOf(2, 2, 2, 2), i = 2, j = 2, k = 2, l = 2)

        assertEquals(expected = 0.0992f, actual = actual[0][0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.6977f, actual = actual[0][0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3182f, actual = actual[0][0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.9840f, actual = actual[0][0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.3500f, actual = actual[0][1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.9797f, actual = actual[0][1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.0691f, actual = actual[0][1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3833f, actual = actual[0][1][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.8967f, actual = actual[1][0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.9967f, actual = actual[1][0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.7926f, actual = actual[1][0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.0967f, actual = actual[1][0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.0361f, actual = actual[1][1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.3765f, actual = actual[1][1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.6351f, actual = actual[1][1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.2895f, actual = actual[1][1][1][1], absoluteTolerance = 1e-4f)
    }
}
