@file:Suppress("NonAsciiCharacters")

package com.wsr.operation

import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.d3
import com.wsr.core.get
import com.wsr.core.operation.div.div
import kotlin.test.Test
import kotlin.test.assertEquals

class DivExtTest {
    @Test
    fun `D1÷Float=各要素をFloatで割ったD1`() {
        val a = IOType.d1(listOf(2.0f, 4.0f, 6.0f))
        val result = a / 2.0f
        assertEquals(
            expected = IOType.d1(listOf(1.0f, 2.0f, 3.0f)),
            actual = result,
        )
    }

    @Test
    fun `Float÷D2=Floatを各要素で割ったD2`() {
        val a = IOType.d2(2, 2) { x, y -> (x * 2 + y + 1).toFloat() }
        val result = 12.0f / a
        assertEquals(expected = 12f, actual = result[0, 0], absoluteTolerance = 1e-5f)
        assertEquals(expected = 6f, actual = result[0, 1], absoluteTolerance = 1e-5f)
        assertEquals(expected = 4f, actual = result[1, 0], absoluteTolerance = 1e-5f)
        assertEquals(expected = 3f, actual = result[1, 1], absoluteTolerance = 1e-5f)
    }

    @Test
    fun `Float÷D3=Floatを各要素で割ったD3`() {
        val a = IOType.d3(2, 2, 2) { x, y, z -> (x * 4 + y * 2 + z + 1).toFloat() }
        val result = 24.0f / a
        assertEquals(expected = 24f, actual = result[0, 0, 0], absoluteTolerance = 1e-5f)
        assertEquals(expected = 12f, actual = result[0, 0, 1], absoluteTolerance = 1e-5f)
        assertEquals(expected = 8f, actual = result[0, 1, 0], absoluteTolerance = 1e-5f)
        assertEquals(expected = 6f, actual = result[0, 1, 1], absoluteTolerance = 1e-5f)
        assertEquals(expected = 4.8f, actual = result[1, 0, 0], absoluteTolerance = 1e-5f)
        assertEquals(expected = 4f, actual = result[1, 0, 1], absoluteTolerance = 1e-5f)
        assertEquals(expected = 3.42857f, actual = result[1, 1, 0], absoluteTolerance = 1e-5f)
        assertEquals(expected = 3f, actual = result[1, 1, 1], absoluteTolerance = 1e-5f)
    }
}
