@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.elementwise.math
import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.ioTypeTestRule
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.math.tanh
import kotlin.test.Test

class D1ExtTest {
    @Test
    fun `exp=D1バッチの指数関数`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d1(listOf(0f, 1f)),
            IOType.d1(listOf(2f, 0f)),
        )
        val result = batch.exp()
        assertContentEquals(IOType.d1(listOf(1f, exp(1f))), result[0])
        assertContentEquals(IOType.d1(listOf(exp(2f), 1f)), result[1])
    }

    @Test
    fun `pow=D1バッチのべき乗`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d1(listOf(2f, 3f)),
            IOType.d1(listOf(4f, 5f)),
        )
        val result = batch.pow(2)
        assertContentEquals(IOType.d1(listOf(4f, 9f)), result[0])
        assertContentEquals(IOType.d1(listOf(16f, 25f)), result[1])
    }

    @Test
    fun `sqrt=D1バッチの平方根`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d1(listOf(4f, 9f)),
            IOType.d1(listOf(16f, 25f)),
        )
        val result = batch.sqrt()
        assertContentEquals(IOType.d1(listOf(2f, 3f)), result[0])
        assertContentEquals(IOType.d1(listOf(4f, 5f)), result[1])
    }

    @Test
    fun `ln=D1バッチの自然対数`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d1(listOf(1f, exp(1f))),
            IOType.d1(listOf(exp(2f), 1f)),
        )
        val result = batch.ln()
        assertContentEquals(IOType.d1(listOf(0f, 1f)), result[0])
        assertContentEquals(IOType.d1(listOf(2f, 0f)), result[1])
    }

    @Test
    fun `tanh=D1バッチの双曲線正接`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d1(listOf(0f, 1f)),
            IOType.d1(listOf(-1f, 2f)),
        )
        val result = batch.tanh()
        assertContentEquals(IOType.d1(listOf(tanh(0f), tanh(1f))), result[0])
        assertContentEquals(IOType.d1(listOf(tanh(-1f), tanh(2f))), result[1])
    }
}
