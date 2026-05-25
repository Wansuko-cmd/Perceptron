@file:Suppress("NonAsciiCharacters")

package com.wsr.core.shape

import com.wsr.assertContentEquals
import com.wsr.core.IOType
import com.wsr.core.d3
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class ShapeD3Test {
    val input = IOType.d3(2, 3, 4) { i, j, k -> i * 12f + j * 4f + k }

    @Test
    fun `flip_axis0=3次元flip`() = ioTypeTestRule {
        val actual = input.flip(axis = 0)
        assertContentEquals(
            expected = IOType.d3(2, 3, 4) { i, j, k -> (1 - i) * 12f + j * 4f + k },
            actual = actual,
        )
    }

    @Test
    fun `flip_axis1=3次元flip`() = ioTypeTestRule {
        val actual = input.flip(axis = 1)
        assertContentEquals(
            expected = IOType.d3(2, 3, 4) { i, j, k -> i * 12f + (2 - j) * 4f + k },
            actual = actual,
        )
    }

    @Test
    fun `flip_axis2=3次元flip`() = ioTypeTestRule {
        val actual = input.flip(axis = 2)
        assertContentEquals(
            expected = IOType.d3(2, 3, 4) { i, j, k -> i * 12f + j * 4f + (3 - k) },
            actual = actual,
        )
    }
}
