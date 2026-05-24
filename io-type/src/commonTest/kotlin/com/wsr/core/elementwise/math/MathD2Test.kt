@file:Suppress("NonAsciiCharacters")

package com.wsr.core.elementwise.math

import com.wsr.assertContentEquals
import com.wsr.core.IOType
import com.wsr.core.d2
import com.wsr.core.elementwise.math.exp
import com.wsr.core.elementwise.math.pow
import com.wsr.core.elementwise.math.softmax
import com.wsr.core.elementwise.math.sqrt
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class MathD2Test {
    @Test
    fun `exp=指数関数`() = ioTypeTestRule {
        val d2 = IOType.d2(1, 1) { _, _ -> 0f }
        val actual = d2.exp()
        assertContentEquals(
            expected = IOType.d2(1, 1) { _, _ -> 1f },
            actual = actual,
        )
    }

    @Test
    fun `pow=階乗`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 2) { i, j -> i * 2f + j + 2f }
        val actual = d2.pow(n = 2)
        assertContentEquals(
            expected = IOType.d2(2, 2) { i, j ->
                val v = i * 2f + j + 2f
                v * v
            },
            actual = actual,
        )
    }

    @Test
    fun `sqrt=平方根`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 2) { i, j -> listOf(4f, 9f, 16f, 25f)[i * 2 + j] }
        val actual = d2.sqrt()
        assertContentEquals(
            expected = IOType.d2(2, 2) { i, j -> listOf(2f, 3f, 4f, 5f)[i * 2 + j] },
            actual = actual,
        )
    }

    @Test
    fun `softmax=ソフトマックス`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 2) { _, _ -> 1f }
        val actual = d2.softmax()
        assertContentEquals(
            expected = IOType.d2(2, 2) { _, _ -> 0.25f },
            actual = actual,
        )
    }

    @Test
    fun `softmax_axis0=列方向ソフトマックス`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 2) { _, _ -> 1f }
        val actual = d2.softmax(axis = 0)
        assertContentEquals(
            expected = IOType.d2(2, 2) { _, _ -> 0.5f },
            actual = actual,
        )
    }

    @Test
    fun `softmax_axis1=行方向ソフトマックス`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { _, _ -> 1f }
        val actual = d2.softmax(axis = 1)
        assertContentEquals(
            expected = IOType.d2(2, 3) { _, _ -> 1f / 3f },
            actual = actual,
        )
    }
}
