@file:Suppress("NonAsciiCharacters")

package com.wsr.core

import kotlin.test.Test
import kotlin.test.assertEquals

class D1Test {
    @Test
    fun `get=1次元要素取得`() {
        val d1 = IOType.d1(listOf(1.0f, 2.0f, 3.0f, 4.0f))
        assertEquals(expected = 1.0f, actual = d1[0])
        assertEquals(expected = 2.0f, actual = d1[1])
        assertEquals(expected = 3.0f, actual = d1[2])
        assertEquals(expected = 4.0f, actual = d1[3])
    }

    @Test
    fun `set=1次元要素設定`() {
        val d1 = IOType.d1(listOf(1.0f, 2.0f, 3.0f, 4.0f))
        d1[0] = 4.0f
        d1[1] = 3.0f
        d1[2] = 2.0f
        d1[3] = 1.0f
        assertEquals(expected = 4.0f, actual = d1[0])
        assertEquals(expected = 3.0f, actual = d1[1])
        assertEquals(expected = 2.0f, actual = d1[2])
        assertEquals(expected = 1.0f, actual = d1[3])
    }
}
