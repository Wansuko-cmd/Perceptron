@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d4
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class ConcatD4Test {
    @Test
    fun `concat_axis0=axis0で4次元を連結`() = ioTypeTestRule {
        val d4a = IOType.d4(1, 2, 2, 2) { _, j, k, l -> j * 4f + k * 2f + l }
        val d4b = IOType.d4(1, 2, 2, 2) { _, j, k, l -> 8f + j * 4f + k * 2f + l }
        val actual = d4a.concat(d4b, axis = 0)
        assertContentEquals(
            expected = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l },
            actual = actual,
        )
    }

    @Test
    fun `concat_axis1=axis1で4次元を連結`() = ioTypeTestRule {
        val d4a = IOType.d4(2, 1, 2, 2) { i, _, k, l -> i * 8f + k * 2f + l }
        val d4b = IOType.d4(2, 1, 2, 2) { i, _, k, l -> i * 8f + k * 2f + l + 4f }
        val actual = d4a.concat(d4b, axis = 1)
        assertContentEquals(
            expected = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l },
            actual = actual,
        )
    }

    @Test
    fun `concat_axis2=axis2で4次元を連結`() = ioTypeTestRule {
        val d4a = IOType.d4(2, 2, 1, 2) { i, j, _, l -> i * 8f + j * 4f + l }
        val d4b = IOType.d4(2, 2, 1, 2) { i, j, _, l -> i * 8f + j * 4f + l + 2f }
        val actual = d4a.concat(d4b, axis = 2)
        assertContentEquals(
            expected = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l },
            actual = actual,
        )
    }

    @Test
    fun `concat_axis3=axis3で4次元を連結`() = ioTypeTestRule {
        val d4a = IOType.d4(2, 2, 2, 1) { i, j, k, _ -> i * 8f + j * 4f + k * 2f }
        val d4b = IOType.d4(2, 2, 2, 1) { i, j, k, _ -> i * 8f + j * 4f + k * 2f + 1f }
        val actual = d4a.concat(d4b, axis = 3)
        assertContentEquals(
            expected = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l },
            actual = actual,
        )
    }
}
