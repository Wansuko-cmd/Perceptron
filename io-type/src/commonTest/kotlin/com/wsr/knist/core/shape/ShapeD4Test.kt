@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core.shape

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d4
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class ShapeD4Test {
    val input = IOType.d4(2, 3, 2, 2) { i, j, k, l -> i * 12f + j * 4f + k * 2f + l }

    @Test
    fun `flip_axis0=4次元flip`() = ioTypeTestRule {
        val actual = input.flip(axis = 0)
        assertContentEquals(
            expected = IOType.d4(2, 3, 2, 2) { i, j, k, l -> (1 - i) * 12f + j * 4f + k * 2f + l },
            actual = actual,
        )
    }

    @Test
    fun `flip_axis1=4次元flip`() = ioTypeTestRule {
        val actual = input.flip(axis = 1)
        assertContentEquals(
            expected = IOType.d4(2, 3, 2, 2) { i, j, k, l -> i * 12f + (2 - j) * 4f + k * 2f + l },
            actual = actual,
        )
    }

    @Test
    fun `flip_axis2=4次元flip`() = ioTypeTestRule {
        val actual = input.flip(axis = 2)
        assertContentEquals(
            expected = IOType.d4(2, 3, 2, 2) { i, j, k, l -> i * 12f + j * 4f + (1 - k) * 2f + l },
            actual = actual,
        )
    }

    @Test
    fun `flip_axis3=4次元flip`() = ioTypeTestRule {
        val actual = input.flip(axis = 3)
        assertContentEquals(
            expected = IOType.d4(2, 3, 2, 2) { i, j, k, l -> i * 12f + j * 4f + k * 2f + (1 - l) },
            actual = actual,
        )
    }
}
