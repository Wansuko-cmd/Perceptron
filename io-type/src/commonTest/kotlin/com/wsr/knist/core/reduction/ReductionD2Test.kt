@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core.reduction

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.reduction.average
import com.wsr.knist.core.reduction.max
import com.wsr.knist.core.reduction.min
import com.wsr.knist.core.reduction.sum
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class ReductionD2Test {
    @Test
    fun `2次元平均`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.average()
        assertContentEquals(
            expected = IOType.d0(2.5f),
            actual = actual,
        )
    }

    @Test
    fun `2次元平均_axis=0`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.average(axis = 0)
        assertContentEquals(
            expected = IOType.d1(listOf(1.5f, 2.5f, 3.5f)),
            actual = actual,
        )
    }

    @Test
    fun `2次元平均_axis=1`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.average(axis = 1)
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 4f)),
            actual = actual,
        )
    }

    @Test
    fun `2次元内最大値`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.max()
        assertContentEquals(
            expected = IOType.d0(5f),
            actual = actual,
        )
    }

    @Test
    fun `2次元最大値_axis=0`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.max(axis = 0)
        assertContentEquals(
            expected = IOType.d1(listOf(3f, 4f, 5f)),
            actual = actual,
        )
    }

    @Test
    fun `2次元最大値_axis=1`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.max(axis = 1)
        assertContentEquals(
            expected = IOType.d1(listOf(2f, 5f)),
            actual = actual,
        )
    }

    @Test
    fun `2次元内最小値`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.min()
        assertContentEquals(
            expected = IOType.d0(0f),
            actual = actual,
        )
    }

    @Test
    fun `2次元最小値_axis=0`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.min(axis = 0)
        assertContentEquals(
            expected = IOType.d1(listOf(0f, 1f, 2f)),
            actual = actual,
        )
    }

    @Test
    fun `2次元最小値_axis=1`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.min(axis = 1)
        assertContentEquals(
            expected = IOType.d1(listOf(0f, 3f)),
            actual = actual,
        )
    }

    @Test
    fun `2次元合計`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.sum()
        assertContentEquals(
            expected = IOType.d0(15f),
            actual = actual,
        )
    }

    @Test
    fun `2次元合計_axis=0`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.sum(axis = 0)
        assertContentEquals(
            expected = IOType.d1(listOf(3f, 5f, 7f)),
            actual = actual,
        )
    }

    @Test
    fun `2次元合計_axis=1`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2.sum(axis = 1)
        assertContentEquals(
            expected = IOType.d1(listOf(3f, 12f)),
            actual = actual,
        )
    }
}
