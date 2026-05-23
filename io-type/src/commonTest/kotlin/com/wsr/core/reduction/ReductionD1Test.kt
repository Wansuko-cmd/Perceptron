@file:Suppress("NonAsciiCharacters")

package com.wsr.core.reduction

import com.wsr.assertContentEquals
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.core.d1
import com.wsr.core.reduction.average
import com.wsr.core.reduction.max
import com.wsr.core.reduction.maxIndex
import com.wsr.core.reduction.min
import com.wsr.core.reduction.sum
import com.wsr.ioTypeTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class ReductionD1Test {
    @Test
    fun `1次元平均`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 2f, 3f, 4f))
        val actual = d1.average()
        assertContentEquals(
            expected = IOType.d0(2.5f),
            actual = actual,
        )
    }

    @Test
    fun `1次元内最大値`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 4f, 2f, 3f))
        val actual = d1.max()
        assertContentEquals(
            expected = IOType.d0(4f),
            actual = actual,
        )
    }

    @Test
    fun `1次元内最小値`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 4f, 2f, 3f))
        val actual = d1.min()
        assertContentEquals(
            expected = IOType.d0(1f),
            actual = actual,
        )
    }

    @Test
    fun `1次元合計`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 2f, 3f, 4f))
        val actual = d1.sum()
        assertContentEquals(
            expected = IOType.d0(10f),
            actual = actual,
        )
    }

    @Test
    fun `1次元最大値インデックス`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 4f, 2f, 3f))
        val actual = d1.maxIndex()
        assertEquals(expected = 1, actual = actual)
    }
}
