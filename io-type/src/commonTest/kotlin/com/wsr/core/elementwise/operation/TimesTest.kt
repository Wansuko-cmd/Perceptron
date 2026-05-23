@file:Suppress("NonAsciiCharacters")

package com.wsr.core.elementwise.operation

import com.wsr.assertContentEquals
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.elementwise.operation.times.times
import com.wsr.core.get
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class TimesTest {
    @Test
    fun `1次元×スカラー`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 2f, 3f, 4f))
        val actual = d1 * 2f
        assertContentEquals(
            expected = IOType.d1(listOf(2f, 4f, 6f, 8f)),
            actual = actual,
        )
    }

    @Test
    fun `1次元×0次元`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(1f, 2f, 3f, 4f))
        val d0 = IOType.d0(2f)
        val actual = d1 * d0
        assertContentEquals(
            expected = IOType.d1(listOf(2f, 4f, 6f, 8f)),
            actual = actual,
        )
    }

    @Test
    fun `1次元×1次元`() = ioTypeTestRule {
        val d1a = IOType.d1(listOf(1f, 2f, 3f, 4f))
        val d1b = IOType.d1(listOf(2f, 3f, 4f, 5f))
        val actual = d1a * d1b
        assertContentEquals(
            expected = IOType.d1(listOf(2f, 6f, 12f, 20f)),
            actual = actual,
        )
    }

    @Test
    fun `2次元×スカラー`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 2) { i, j -> i * 2f + j + 1f }
        val actual = d2 * 2f
        assertContentEquals(
            expected = IOType.d2(2, 2) { i, j -> (i * 2f + j + 1f) * 2f },
            actual = actual,
        )
    }

    @Test
    fun `2次元×2次元`() = ioTypeTestRule {
        val d2a = IOType.d2(2, 2) { i, j -> i * 2f + j + 1f }
        val d2b = IOType.d2(2, 2) { _, _ -> 2f }
        val actual = d2a * d2b
        assertContentEquals(
            expected = IOType.d2(2, 2) { i, j -> (i * 2f + j + 1f) * 2f },
            actual = actual,
        )
    }

    @Test
    fun `2次元_axis=0×1次元`() = ioTypeTestRule {
        val d2 = IOType.d2(3, 4) { i, j -> i * 4f + j }
        val d1 = IOType.d1(listOf(1f, 2f, 3f))
        val actual = d2.times(d1, axis = 0)
        assertContentEquals(
            expected = IOType.d2(3, 4) { i, j -> (i * 4f + j) * (i + 1f) },
            actual = actual,
        )
    }

    @Test
    fun `2次元_axis=1×1次元`() = ioTypeTestRule {
        val d2 = IOType.d2(3, 4) { i, j -> i * 4f + j }
        val d1 = IOType.d1(listOf(1f, 2f, 3f, 4f))
        val actual = d2.times(d1, axis = 1)
        assertContentEquals(
            expected = IOType.d2(3, 4) { i, j -> (i * 4f + j) * (j + 1f) },
            actual = actual,
        )
    }
}
