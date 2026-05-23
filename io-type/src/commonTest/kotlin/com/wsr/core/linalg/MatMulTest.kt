@file:Suppress("NonAsciiCharacters")

package com.wsr.core.linalg

import com.wsr.assertContentEquals
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.linalg.matMul
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class MatMulTest {
    @Test
    fun `D2 matMul D1=行列積`() = ioTypeTestRule {
        val d2 = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val d1 = IOType.d1(listOf(1f, 2f, 3f))
        val actual = d2.matMul(d1)
        assertContentEquals(
            expected = IOType.d1(listOf(8f, 26f)),
            actual = actual,
        )
    }

    @Test
    fun `D2T matMul D1=行列積`() = ioTypeTestRule {
        val d2 = IOType.d2(3, 2) { i, j -> i * 2f + j }
        val d1 = IOType.d1(listOf(1f, 2f, 3f))
        val actual = d2.matMul(d1, trans = true)
        assertContentEquals(
            expected = IOType.d1(listOf(16f, 22f)),
            actual = actual,
        )
    }

    @Test
    fun `D2 matMul D2=行列積`() = ioTypeTestRule {
        val d2a = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val d2b = IOType.d2(3, 2) { i, j -> i * 2f + j }
        val actual = d2a.matMul(d2b)
        assertContentEquals(
            expected = IOType.d2(listOf(2, 2), listOf(10f, 13f, 28f, 40f)),
            actual = actual,
        )
    }

    @Test
    fun `D2 matMul D2T=行列積`() = ioTypeTestRule {
        val d2a = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val d2b = IOType.d2(2, 3) { i, j -> i * 3f + j }
        val actual = d2a.matMul(d2b, transB = true)
        assertContentEquals(
            expected = IOType.d2(listOf(2, 2), listOf(5f, 14f, 14f, 50f)),
            actual = actual,
        )
    }
}
