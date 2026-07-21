@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.bias.d3

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.process.GraphEnv
import kotlin.test.Test

class BiasD3Test {
    val target
        get() = BiasD3(
            inputI = 2,
            inputJ = 2,
            inputK = 2,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d3(2, 2, 2),
            weight = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
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
    fun `expect=バイアス項`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 3f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(2f, 6f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(4f, 3f), actual = actual[0][1][0])
        assertContentEquals(expected = IOType.d1(6f, 6f), actual = actual[0][1][1])
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 3f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(2f, 6f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(4f, 3f), actual = actual[0][1][0])
        assertContentEquals(expected = IOType.d1(6f, 6f), actual = actual[0][1][1])
    }

    @Test
    fun `train=重みを更新する`() = networkScopeTestRule {
        val target = target

        with(target) { _train(input = input, env = GraphEnv(), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 2.97f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(1.98f, 5.94f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(3.96f, 2.9699f), actual = actual[0][1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(5.94f, 5.94f), actual = actual[0][1][1])
    }
}
