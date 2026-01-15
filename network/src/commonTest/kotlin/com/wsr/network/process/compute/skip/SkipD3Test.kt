@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.skip

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.d3
import com.wsr.core.get
import com.wsr.network.networkTestRule
import com.wsr.network.optimizer.Scheduler
import com.wsr.network.optimizer.sgd.Sgd
import com.wsr.network.process.Context
import com.wsr.network.process.compute.bias.d3.BiasD3
import kotlin.test.Test
import kotlin.test.assertEquals

class SkipD3Test {
    val target = SkipD3(
        layers = listOf(
            BiasD3(
                outputX = 2,
                outputY = 2,
                outputZ = 3,
                optimizer = Sgd(Scheduler.Fix(0.01f)).d3(2, 2, 2),
                weight = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            ),
            BiasD3(
                outputX = 2,
                outputY = 2,
                outputZ = 3,
                optimizer = Sgd(Scheduler.Fix(0.01f)).d3(2, 2, 2),
                weight = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            ),
        ),
        inputX = 2,
        inputY = 2,
        inputZ = 2,
        outputX = 2,
        outputY = 2,
        outputZ = 2,
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
    fun `expect=スキップ接続を行う`() = networkTestRule {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 6f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(4f, 12f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(8f, 6f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(12f, 12f), actual = actual[0][1][1])
    }

    @Test
    fun `train=勾配を伝播`() = networkTestRule {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 12f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(8f, 24f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(16f, 12f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(24f, 24f), actual = actual[0][1][1])
    }
}
