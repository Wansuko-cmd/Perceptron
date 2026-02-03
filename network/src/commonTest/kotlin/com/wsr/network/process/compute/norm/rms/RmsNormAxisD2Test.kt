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
import com.wsr.network.process.compute.norm.rms.d2.RmsNormAxisD2
import kotlin.test.Test

class RmsNormAxisD2Test {
    val target0 get() = RmsNormAxisD2(outputX = 2, outputY = 3, axis = 0, e = 1e-6f)
    val target1 get() = RmsNormAxisD2(outputX = 2, outputY = 3, axis = 1, e = 1e-6f)
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
    fun `Axis0_expect=axis0で層正規化`() = networkTestRule {
        val actual = target0._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(
            expected = IOType.d1(0.0000f, 0.6324f, 0.6324f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 1.2649f, 1.2649f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 1.2126f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 1.4141f, 0.7276f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis0_train=axis0で正規化および勾配を伝播`() = networkTestRule {
        val actual = target0._train(
            input = input,
            context = Context(input),
            calcDelta = { 1e6f * it as Batch<IOType.D2> },
        ) as Batch<IOType.D2>

        assertEquals(
            expected = IOType.d1(0.0000f, 0.1250f, 0.0000f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 0.2500f, 0.0000f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 2.2500f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 147.5000f, 1.3125f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis1_expect=axis1で層正規化`() = networkTestRule {
        val actual = target1._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(
            expected = IOType.d1(0.0000f, 0.7745f, 1.5491f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 0.7745f, 1.5491f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 1.7320f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 0.7745f, 1.5491f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis1_train=axis1で正規化および勾配を伝播`() = networkTestRule {
        val actual = target1._train(
            input = input,
            context = Context(input),
            calcDelta = { 1e6f * it as Batch<IOType.D2> },
        ) as Batch<IOType.D2>

        assertEquals(
            expected = IOType.d1(0.0000f, 0.3125f, 0.6250f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 0.0312f, 0.0625f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 9.0000f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(0.0000f, 13.1250f, 26.2500f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
