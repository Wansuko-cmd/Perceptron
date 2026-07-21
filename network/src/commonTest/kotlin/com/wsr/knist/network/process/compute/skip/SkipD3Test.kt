@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.skip

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.GraphBuilder
import com.wsr.knist.network.GraphId
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.process.GraphEnv
import com.wsr.knist.network.process.compute.bias.d3.BiasD3
import com.wsr.knist.network.process.reshape.reshape.reshapeToD1
import com.wsr.knist.network.process.reshape.reshape.reshapeToD3
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

class SkipD3Test {
    val target = SkipD3(
        layers = listOf(
            BiasD3(
                inputI = 2,
                inputJ = 2,
                inputK = 3,
                optimizer = Sgd(Scheduler.Fix(0.01f)).d3(2, 2, 2),
                weight = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            ),
            BiasD3(
                inputI = 2,
                inputJ = 2,
                inputK = 3,
                optimizer = Sgd(Scheduler.Fix(0.01f)).d3(2, 2, 2),
                weight = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            ),
        ),
        inputI = 2,
        inputJ = 2,
        inputK = 2,
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
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 6f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(4f, 12f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(8f, 6f), actual = actual[0][1][0])
        assertContentEquals(expected = IOType.d1(12f, 12f), actual = actual[0][1][1])
    }

    @Test
    fun `train=勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 12f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(8f, 24f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(16f, 12f), actual = actual[0][1][0])
        assertContentEquals(expected = IOType.d1(24f, 24f), actual = actual[0][1][1])
    }

    private fun builder() = GraphBuilder.Node.D3(
        inputI = 2,
        inputJ = 3,
        inputK = 4,
        from = GraphId(),
        nodes = emptyList(),
        optimizer = Sgd(Scheduler.Fix(rate = 0.01f)),
        initializer = Fixed(0.5f),
    )

    @Test
    fun `skip=空ブロックは何も追加しない`() {
        val builder = builder()

        val actual = builder.skip { this }

        assertSame(builder, actual)
    }

    @Test
    fun `skip=出力形状が入力と異なる場合は例外を投げる`() {
        val builder = builder()

        assertFailsWith<IllegalStateException> {
            builder.skip { this.reshapeToD1().reshapeToD3(i = 4, j = 3, k = 2) }
        }
    }
}
