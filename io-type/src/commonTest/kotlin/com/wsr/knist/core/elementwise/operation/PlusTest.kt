@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core.elementwise.operation

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.elementwise.operation.plus.plus
import com.wsr.knist.core.get
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class PlusTest {
    @Test
    fun `1次元+スカラー`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(0f, 1f, 2f, 3f))
        val actual = d1 + 1f
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 2f, 3f, 4f)),
            actual = actual,
        )
    }

    @Test
    fun `1次元+0次元`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(0f, 1f, 2f, 3f))
        val d0 = IOType.d0(1f)
        val actual = d1 + d0
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 2f, 3f, 4f)),
            actual = actual,
        )
    }

    @Test
    fun `1次元+1次元`() = ioTypeTestRule {
        val d1a = IOType.d1(listOf(0f, 1f, 2f, 3f))
        val d1b = IOType.d1(listOf(1f, 2f, 3f, 4f))
        val actual = d1a + d1b
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 3f, 5f, 7f)),
            actual = actual,
        )
    }

    @Test
    fun `1次元+2次元_axis=0`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(0f, 1f, 2f))
        val d2 = IOType.d2(3, 4) { i, j -> i * 4f + j }
        val actual = d1.plus(d2, axis = 0)
        assertContentEquals(
            expected = IOType.d2(3, 4) { i, j -> i * 4f + j + i },
            actual = actual,
        )
    }

    @Test
    fun `1次元+2次元_axis=1`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(0f, 1f, 2f, 3f))
        val d2 = IOType.d2(3, 4) { i, j -> i * 4f + j }
        val actual = d1.plus(d2, axis = 1)
        assertContentEquals(
            expected = IOType.d2(3, 4) { i, j -> i * 4f + j + j },
            actual = actual,
        )
    }

    @Test
    fun `2次元+スカラー`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val actual = d2 + 1f
        assertContentEquals(
            expected = IOType.d2(2, 2) { i, j -> i * 2f + j + 1f },
            actual = actual,
        )
    }

    @Test
    fun `2次元+2次元`() = ioTypeTestRule {
        val d2a = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val d2b = IOType.d2(2, 2) { i, j -> 4f + i * 2f + j }
        val actual = d2a + d2b
        assertContentEquals(
            expected = IOType.d2(2, 2) { i, j -> 4f + (i * 2f + j) * 2f },
            actual = actual,
        )
    }

    @Test
    fun `2次元_axis=0+1次元`() = ioTypeTestRule {
        val d2 = IOType.d2(3, 4) { i, j -> i * 4f + j }
        val d1 = IOType.d1(listOf(0f, 1f, 2f))
        val actual = d2.plus(d1, axis = 0)
        assertContentEquals(
            expected = IOType.d2(3, 4) { i, j -> i * 4f + j + i },
            actual = actual,
        )
    }

    @Test
    fun `2次元_axis=1+1次元`() = ioTypeTestRule {
        val d2 = IOType.d2(3, 4) { i, j -> i * 4f + j }
        val d1 = IOType.d1(listOf(0f, 1f, 2f, 3f))
        val actual = d2.plus(d1, axis = 1)
        assertContentEquals(
            expected = IOType.d2(3, 4) { i, j -> i * 4f + j + j },
            actual = actual,
        )
    }
}
