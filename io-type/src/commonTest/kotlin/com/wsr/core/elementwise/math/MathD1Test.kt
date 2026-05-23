@file:Suppress("NonAsciiCharacters")

package com.wsr.core.elementwise.math

import com.wsr.assertContentEquals
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.elementwise.math.exp
import com.wsr.core.elementwise.math.ln
import com.wsr.core.elementwise.math.pow
import com.wsr.core.elementwise.math.softmax
import com.wsr.core.elementwise.math.sqrt
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class MathD1Test {
    @Test
    fun `exp=指数関数`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(0f))
        val actual = d1.exp()
        assertContentEquals(
            expected = IOType.d1(listOf(1f)),
            actual = actual,
        )
    }

    @Test
    fun `ln=自然対数`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f))
        val actual = d1.ln(e = 1e-7f)
        assertContentEquals(
            expected = IOType.d1(listOf(0f)),
            actual = actual,
        )
    }

    @Test
    fun `pow=階乗`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(2f, 3f, 4f))
        val actual = d1.pow(n = 2)
        assertContentEquals(
            expected = IOType.d1(listOf(4f, 9f, 16f)),
            actual = actual,
        )
    }

    @Test
    fun `sqrt=平方根`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(4f, 9f, 16f))
        val actual = d1.sqrt()
        assertContentEquals(
            expected = IOType.d1(listOf(2f, 3f, 4f)),
            actual = actual,
        )
    }

    @Test
    fun `softmax=ソフトマックス`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 1f, 1f))
        val actual = d1.softmax()
        assertContentEquals(
            expected = IOType.d1(listOf(1f / 3f, 1f / 3f, 1f / 3f)),
            actual = actual,
        )
    }
}
