@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.norm.layer

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.elementwise.operation.times.times
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.network.assertContentEquals
import com.wsr.network.networkTestRule
import com.wsr.network.process.Context
import com.wsr.network.process.compute.norm.layer.d1.LayerNormD1
import kotlin.test.Test

class LayerNormD1Test {
    val target get() = LayerNormD1(outputSize = 3, e = 1e-6f)
    val input
        get() = batchOf(
            IOType.d1(3) { it.toFloat() },
            IOType.d1(3) { it * 2f },
        )

    @Test
    fun `expect=層正規化`() = networkTestRule {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D1>

        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0f, 1.2247f),
            actual = actual[0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0f, 1.2247f),
            actual = actual[1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=正規化および勾配を伝播`() = networkTestRule {
        val actual = target._train(
            input = input,
            context = Context(input),
            calcDelta = { 1e6f * it as Batch<IOType.D1> },
        ) as Batch<IOType.D1>

        assertContentEquals(
            expected = IOType.d1(-2.2500f, 0.0000f, 2.2500f),
            actual = actual[0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.1875f, 0.0000f, 0.1875f),
            actual = actual[1],
            absoluteTolerance = 1e-4f,
        )
    }
}
