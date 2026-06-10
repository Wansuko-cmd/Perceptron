@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core.reduction

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.core.reduction.average
import com.wsr.knist.core.reduction.max
import com.wsr.knist.core.reduction.maxIndex
import com.wsr.knist.core.reduction.min
import com.wsr.knist.core.reduction.sum
import com.wsr.knist.ioTypeTestRule
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
        assertEquals(expected = IOType.d0(1f), actual = actual)
    }
}
