@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core.reduction

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.core.reduction.maxIndex
import com.wsr.knist.core.reduction.topK
import com.wsr.knist.core.reduction.topP
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class ReductionD4Test {

    private val d4 = IOType.d4(
        shape = listOf(2, 2, 2, 2),
        value = listOf(
            1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,
            9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f,
        ),
    )

    private fun allOnes() = IOType.d3(
        shape = listOf(2, 2, 2),
        value = List(8) { 1f },
    )

    @Test
    fun `4次元maxIndex_axis=0`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.maxIndex(axis = 0))
    }

    @Test
    fun `4次元maxIndex_axis=1`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.maxIndex(axis = 1))
    }

    @Test
    fun `4次元maxIndex_axis=2`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.maxIndex(axis = 2))
    }

    @Test
    fun `4次元maxIndex_axis=3`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.maxIndex(axis = 3))
    }

    @Test
    fun `4次元topK_k=1_axis=0`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.topK(k = 1, axis = 0))
    }

    @Test
    fun `4次元topK_k=1_axis=1`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.topK(k = 1, axis = 1))
    }

    @Test
    fun `4次元topK_k=1_axis=2`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.topK(k = 1, axis = 2))
    }

    @Test
    fun `4次元topK_k=1_axis=3`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.topK(k = 1, axis = 3))
    }

    @Test
    fun `4次元topP_p=0_axis=0`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.topP(p = 0f, axis = 0))
    }

    @Test
    fun `4次元topP_p=0_axis=1`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.topP(p = 0f, axis = 1))
    }

    @Test
    fun `4次元topP_p=0_axis=2`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.topP(p = 0f, axis = 2))
    }

    @Test
    fun `4次元topP_p=0_axis=3`() = ioTypeTestRule {
        assertContentEquals(expected = allOnes(), actual = d4.topP(p = 0f, axis = 3))
    }
}
