@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.skip

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.GraphBuilder
import com.wsr.knist.network.GraphId
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.process.compute.affine.affine
import com.wsr.knist.network.process.compute.bias.d1.BiasD1
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

class SkipD1Test {
    val target = SkipD1(
        layers = listOf(
            BiasD1(
                inputI = 3,
                optimizer = Sgd(Scheduler.Fix(rate = 0.01f)).d1(3),
                weight = IOType.d1(3) { it.toFloat() },
            ),
            BiasD1(
                inputI = 3,
                optimizer = Sgd(Scheduler.Fix(rate = 0.01f)).d1(3),
                weight = IOType.d1(3) { it.toFloat() },
            ),
        ),
        inputI = 3,
    )
    val input
        get() = Batch.of(
            IOType.d1(3) { it * 2f },
            IOType.d1(3) { it * 3f },
        )

    @Test
    fun `expect=スキップ接続を行う`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d1(0f, 6f, 12f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(0f, 8f, 16f), actual = actual[1])
    }

    @Test
    fun `train=勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d1(0f, 12f, 24f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(0f, 16f, 32f), actual = actual[1])
    }

    private fun builder() = GraphBuilder.Node.D1(
        inputI = 3,
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
            builder.skip { this.affine(neuron = 5) }
        }
    }
}
