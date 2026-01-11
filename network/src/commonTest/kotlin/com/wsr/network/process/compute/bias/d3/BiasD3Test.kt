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
import com.wsr.network.NetworkTestRule
import com.wsr.network.assertEquals
import com.wsr.optimizer.Scheduler
import com.wsr.optimizer.sgd.Sgd
import com.wsr.process.Context
import com.wsr.process.compute.bias.d3.BiasD3
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class BiasD3Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target
        get() = BiasD3(
            outputX = 2,
            outputY = 2,
            outputZ = 2,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d3(2, 2, 2),
            weight = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
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
    fun `expect=バイアス項`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 3f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(2f, 6f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(4f, 3f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(6f, 6f), actual = actual[0][1][1])
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 3f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(2f, 6f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(4f, 3f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(6f, 6f), actual = actual[0][1][1])
    }

    @Test
    fun `train=重みを更新する`() {
        val target = target

        target._train(input = input, context = Context(input), calcDelta = { it })
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 2.97f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(1.98f, 5.94f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(3.96f, 2.9699f), actual = actual[0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = IOType.d1(5.94f, 5.94f), actual = actual[0][1][1])
    }
}
