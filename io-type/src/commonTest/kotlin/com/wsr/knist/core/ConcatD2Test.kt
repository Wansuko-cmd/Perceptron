@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class ConcatD2Test {
    @Test
    fun `concat_axis0=axis0で2次元を連結`() = ioTypeTestRule {
        val d2a = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val d2b = IOType.d2(1, 3) { _, j -> 6f + j }
        val actual = d2a.concat(d2b, axis = 0)
        assertContentEquals(
            expected = IOType.d2(3, 3) { i, j -> i * 3f + j },
            actual = actual,
        )
    }

    @Test
    fun `concat_axis1=axis1で2次元を連結`() = ioTypeTestRule {
        val d2a = IOType.d2(2, 2) { i, j -> i * 4f + j }
        val d2b = IOType.d2(2, 2) { i, j -> i * 4f + j + 2f }
        val actual = d2a.concat(d2b, axis = 1)
        assertContentEquals(
            expected = IOType.d2(2, 4) { i, j -> i * 4f + j },
            actual = actual,
        )
    }
}
