@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core.reduction

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.reduction.maxIndex
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class ReductionD3Test {
    private val d3 = IOType.d3(
        shape = listOf(2, 3, 2),
        value = listOf(
            1f, 2f,
            5f, 1f,
            0f, 3f,

            4f, 0f,
            2f, 4f,
            3f, 1f,
        ),
    )

    @Test
    fun `3次元最大値インデックス_axis=0`() = ioTypeTestRule {
        val actual = d3.maxIndex(axis = 0)
        val exp = floatArrayOf(1f, 0f, 0f, 1f, 1f, 0f)
        assertContentEquals(
            expected = IOType.d2(3, 2) { i, j -> exp[i * 2 + j] },
            actual = actual,
        )
    }

    @Test
    fun `3次元最大値インデックス_axis=1`() = ioTypeTestRule {
        val actual = d3.maxIndex(axis = 1)
        val exp = floatArrayOf(1f, 2f, 0f, 1f)
        assertContentEquals(
            expected = IOType.d2(2, 2) { i, j -> exp[i * 2 + j] },
            actual = actual,
        )
    }

    @Test
    fun `3次元最大値インデックス_axis=2`() = ioTypeTestRule {
        val actual = d3.maxIndex(axis = 2)
        val exp = floatArrayOf(1f, 0f, 1f, 0f, 1f, 0f)
        assertContentEquals(
            expected = IOType.d2(2, 3) { i, j -> exp[i * 3 + j] },
            actual = actual,
        )
    }
}
