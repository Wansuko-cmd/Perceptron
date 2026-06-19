@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class ConcatD1Test {
    @Test
    fun `concat=2つの1次元を連結`() = ioTypeTestRule {
        val d1a = IOType.d1(listOf(1f, 2f, 3f))
        val d1b = IOType.d1(listOf(4f, 5f))
        val actual = d1a.concat(d1b)
        assertContentEquals(
            expected = IOType.d1(listOf(1f, 2f, 3f, 4f, 5f)),
            actual = actual,
        )
    }
}
