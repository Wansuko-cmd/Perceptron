@file:Suppress("NonAsciiCharacters")

package com.wsr.core.elementwise.compare

import com.wsr.assertContentEquals
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.elementwise.compare.gt
import com.wsr.core.elementwise.compare.where.where
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class WhereTest {
    @Test
    fun `Float_where_Float=conditionが1の時x_0のときy`() = ioTypeTestRule {
        val condition = IOType.d1(listOf(1f, 0f, 1f))
        val actual = where(condition, 10f, 0f)
        assertContentEquals(
            expected = IOType.d1(listOf(10f, 0f, 10f)),
            actual = actual,
        )
    }

    @Test
    fun `D1_where_Float=conditionが1の時x_0のときy`() = ioTypeTestRule {
        val condition = IOType.d1(listOf(1f, 0f, 1f))
        val onTrue = IOType.d1(listOf(1f, 2f, 3f))
        val actual = where(condition, onTrue, 0f)
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 0f, 3f)),
            actual = actual,
        )
    }

    @Test
    fun `Float_where_D1=conditionが1の時x_0のときy`() = ioTypeTestRule {
        val condition = IOType.d1(listOf(1f, 0f, 1f))
        val onFalse = IOType.d1(listOf(4f, 5f, 6f))
        val actual = where(condition, 10f, onFalse)
        assertContentEquals(
            expected = IOType.d1(listOf(10f, 5f, 10f)),
            actual = actual,
        )
    }

    @Test
    fun `D1_where_D1=conditionが1の時x_0のときy`() = ioTypeTestRule {
        val condition = IOType.d1(listOf(1f, 0f, 1f))
        val onTrue = IOType.d1(listOf(1f, 2f, 3f))
        val onFalse = IOType.d1(listOf(4f, 5f, 6f))
        val actual = where(condition, onTrue, onFalse)
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 5f, 3f)),
            actual = actual,
        )
    }

    @Test
    fun `D1_where_Float_ラムダ=conditionが1の時x_0のときy`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, -2f, 3f))
        val actual = d1.where(onFalse = 0f) { it gt 0f }
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 0f, 3f)),
            actual = actual,
        )
    }
}
