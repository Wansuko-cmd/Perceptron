@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.elementwise.math

import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.batchOf
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.ioTypeTestRule
import kotlin.math.E
import kotlin.test.Test

class D0ExtTest {
    @Test
    fun `pow=D0バッチのべき乗`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(2f), IOType.d0(3f), IOType.d0(4f))
        val result = batch.pow(2)
        assertContentEquals(IOType.d0(4f), result[0])
        assertContentEquals(IOType.d0(9f), result[1])
        assertContentEquals(IOType.d0(16f), result[2])
    }

    @Test
    fun `sqrt=D0バッチの平方根`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(4f), IOType.d0(9f), IOType.d0(16f))
        val result = batch.sqrt()
        assertContentEquals(IOType.d0(2f), result[0])
        assertContentEquals(IOType.d0(3f), result[1])
        assertContentEquals(IOType.d0(4f), result[2])
    }

    @Test
    fun `ln=D0バッチの自然対数`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(1f), IOType.d0(E.toFloat()))
        val result = batch.ln()
        assertContentEquals(IOType.d0(0f), result[0])
        assertContentEquals(IOType.d0(1f), result[1])
    }
}
