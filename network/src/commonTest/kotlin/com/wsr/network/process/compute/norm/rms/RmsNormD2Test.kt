@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.norm.rms

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.batch.operation.times.times
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.assertEquals
import com.wsr.network.networkTestRule
import com.wsr.network.process.Context
import com.wsr.network.process.compute.norm.rms.d2.RmsNormD2
import kotlin.test.Test

class RmsNormD2Test {
    val target get() = RmsNormD2(outputX = 2, outputY = 3, e = 1e-6f)
    val input
        get() = batchOf(
            IOType.d2(
                IOType.d1(3) { it.toFloat() },
                IOType.d1(3) { it * 2f },
            ),
            IOType.d2(
                IOType.d1(3) { 10f % (it + 1) },
                IOType.d1(3) { it * 0.3f },
            ),
        )

    @Test
    fun `expect=層正規化`() = networkTestRule {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(
            expected = IOType.d1(0.0000f, 0.4898f, 0.9797f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 0.9797f, 1.9595f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 2.0341f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 0.6102f, 1.2205f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=正規化および勾配を伝播`() = networkTestRule {
        val actual = target._train(
            input = input,
            context = Context(input),
            calcDelta = { 1e6f * it as Batch<IOType.D2> },
        ) as Batch<IOType.D2>

        assertEquals(
            expected = IOType.d1(0.0000f, 0.0468f, 0.0937f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 0.0937f, 0.1875f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 17.0000f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 5.0000f, 10.0000f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
