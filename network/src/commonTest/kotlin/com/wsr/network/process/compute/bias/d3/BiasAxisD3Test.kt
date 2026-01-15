@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.bias.d3

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.d3
import com.wsr.core.get
import com.wsr.network.assertEquals
import com.wsr.network.networkTestRule
import com.wsr.network.optimizer.Scheduler
import com.wsr.network.optimizer.sgd.Sgd
import com.wsr.network.process.Context
import kotlin.test.Test
import kotlin.test.assertEquals

class BiasAxisD3Test {
    val target0
        get() = BiasAxisD3(
            outputX = 2,
            outputY = 2,
            outputZ = 2,
            axis = 0,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d1(2),
            weight = IOType.d1(2) { it.toFloat() },
        )

    val target1
        get() = BiasAxisD3(
            outputX = 2,
            outputY = 2,
            outputZ = 2,
            axis = 1,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d1(2),
            weight = IOType.d1(2) { it.toFloat() },
        )

    val target2
        get() = BiasAxisD3(
            outputX = 2,
            outputY = 2,
            outputZ = 2,
            axis = 2,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d1(2),
            weight = IOType.d1(2) { it.toFloat() },
        )

    val input
        get() = batchOf(
            IOType.d3(
                IOType.d2(
                    IOType.d1(2) { it * 2f },
                    IOType.d1(2) { it * 3f },
                ),
                IOType.d2(
                    IOType.d1(2) { it * -2f },
                    IOType.d1(2) { it * -1f },
                ),
            ),
        )

    @Test
    fun `Axis0_expect=axis0で共通のバイアス項`() = networkTestRule {
        val actual = target0._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(0f, 3f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(1f, -1f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(1f, 0f), actual = actual[0][1][1])
    }

    @Test
    fun `Axis0_train=Axis0で共通の勾配を伝播`() = networkTestRule {
        val actual = target0._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(0f, 3f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(1f, -1f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(1f, 0f), actual = actual[0][1][1])
    }

    @Test
    fun `Axis0_train=重みを更新する`() = networkTestRule {
        val target = target0

        target._train(input = input, context = Context(input), calcDelta = { it })
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(
            expected = IOType.d1(-0.0499f, 1.95f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(-0.0499f, 2.95f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(expected = IOType.d1(0.99f, -1.01f), actual = actual[0][1][0])
        assertEquals(
            expected = IOType.d1(0.99f, -0.0099f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis1_expect=axis1で共通のバイアス項`() = networkTestRule {
        val actual = target1._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(1f, 4f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(0f, -2f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(1f, 0f), actual = actual[0][1][1])
    }

    @Test
    fun `Axis1_train=Axis1で共通の勾配を伝播`() = networkTestRule {
        val actual = target1._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(1f, 4f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(0f, -2f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(1f, 0f), actual = actual[0][1][1])
    }

    @Test
    fun `Axis1_train=重みを更新する`() = networkTestRule {
        val target = target1

        target._train(input = input, context = Context(input), calcDelta = { it })
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(0.94f, 3.94f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(0f, -2f), actual = actual[0][1][0])
        assertEquals(
            expected = IOType.d1(0.94f, -0.0600f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis2_expect=axis2で共通のバイアス項`() = networkTestRule {
        val actual = target2._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 3f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(0f, 4f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(0f, -1f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(0f, 0f), actual = actual[0][1][1])
    }

    @Test
    fun `Axis2_train=Axis2で共通の勾配を伝播`() = networkTestRule {
        val actual = target2._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 3f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(0f, 4f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(0f, -1f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(0f, 0f), actual = actual[0][1][1])
    }

    @Test
    fun `Axis2_train=重みを更新する`() = networkTestRule {
        val target = target2

        target._train(input = input, context = Context(input), calcDelta = { it })
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 2.94f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(0f, 3.94f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(0f, -1.06f), actual = actual[0][1][0])
        assertEquals(
            expected = IOType.d1(0f, -0.0600f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
