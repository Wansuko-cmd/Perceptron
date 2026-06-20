@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class ConcatD3Test {
    @Test
    fun `concat_axis0=axis0で3次元を連結`() = ioTypeTestRule {
        val d3a = IOType.d3(1, 2, 3) { _, j, k -> j * 3f + k }
        val d3b = IOType.d3(1, 2, 3) { _, j, k -> 6f + j * 3f + k }
        val actual = d3a.concat(d3b, axis = 0)
        assertContentEquals(
            expected = IOType.d3(2, 2, 3) { i, j, k -> i * 6f + j * 3f + k },
            actual = actual,
        )
    }

    @Test
    fun `concat_axis1=axis1で3次元を連結`() = ioTypeTestRule {
        val d3a = IOType.d3(2, 1, 3) { i, _, k -> i * 6f + k }
        val d3b = IOType.d3(2, 1, 3) { i, _, k -> i * 6f + k + 3f }
        val actual = d3a.concat(d3b, axis = 1)
        assertContentEquals(
            expected = IOType.d3(2, 2, 3) { i, j, k -> i * 6f + j * 3f + k },
            actual = actual,
        )
    }

    @Test
    fun `concat_axis2=axis2で3次元を連結`() = ioTypeTestRule {
        val d3a = IOType.d3(2, 2, 1) { i, j, _ -> i * 4f + j * 2f }
        val d3b = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k + 1f }
        val actual = d3a.concat(d3b, axis = 2)
        assertContentEquals(
            expected = IOType.d3(2, 2, 3) { i, j, k -> i * 4f + j * 2f + k },
            actual = actual,
        )
    }
}
