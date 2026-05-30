@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core.elementwise.compare

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.elementwise.compare.eq
import com.wsr.knist.core.elementwise.compare.gt
import com.wsr.knist.core.elementwise.compare.lt
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class CompareTest {
    @Test
    fun `equals=入力のFloat値と等価かどうか`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 2f, 3f))
        val actual = d1 eq 2f
        assertContentEquals(
            expected = IOType.d1(listOf(0f, 1f, 0f)),
            actual = actual,
        )
    }

    @Test
    fun `equals=入力のD1と等価かどうか`() = ioTypeTestRule {
        val d1a = IOType.d1(listOf(1f, 2f, 3f))
        val d1b = IOType.d1(listOf(1f, 0f, 3f))
        val actual = d1a eq d1b
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 0f, 1f)),
            actual = actual,
        )
    }

    @Test
    fun `greaterThan=入力のFloat値より大きいかどうか`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 2f, 3f))
        val actual = d1 gt 2f
        assertContentEquals(
            expected = IOType.d1(listOf(0f, 0f, 1f)),
            actual = actual,
        )
    }

    @Test
    fun `greaterThan=入力のD1より大きいかどうか`() = ioTypeTestRule {
        val d1a = IOType.d1(listOf(1f, 2f, 3f))
        val d1b = IOType.d1(listOf(2f, 2f, 2f))
        val actual = d1a gt d1b
        assertContentEquals(
            expected = IOType.d1(listOf(0f, 0f, 1f)),
            actual = actual,
        )
    }

    @Test
    fun `lessThan=入力のFloat値より小さいかどうか`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 2f, 3f))
        val actual = d1 lt 2f
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 0f, 0f)),
            actual = actual,
        )
    }

    @Test
    fun `lessThan=入力のD1より小さいかどうか`() = ioTypeTestRule {
        val d1a = IOType.d1(listOf(1f, 2f, 3f))
        val d1b = IOType.d1(listOf(2f, 2f, 2f))
        val actual = d1a lt d1b
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 0f, 0f)),
            actual = actual,
        )
    }
}
