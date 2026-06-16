@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.skip

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.bias.d3.BiasD3
import kotlin.test.Test

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
        outputX = 2,
        outputY = 2,
        outputZ = 2,
    )
    val input
        get() = Batch.of(
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
    fun `expect=スキップ接続を行う`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 6f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(4f, 12f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(8f, 6f), actual = actual[0][1][0])
        assertContentEquals(expected = IOType.d1(12f, 12f), actual = actual[0][1][1])
    }

    @Test
    fun `train=勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 12f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(8f, 24f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(16f, 12f), actual = actual[0][1][0])
        assertContentEquals(expected = IOType.d1(24f, 24f), actual = actual[0][1][1])
    }
}
