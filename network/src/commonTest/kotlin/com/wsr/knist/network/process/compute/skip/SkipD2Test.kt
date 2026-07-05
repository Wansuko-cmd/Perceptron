@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.skip

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.converter.raw.RawD2
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.affine.affine
import com.wsr.knist.network.process.compute.bias.d2.BiasD2
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

class SkipD2Test {
    val target = SkipD2(
        layers = listOf(
            BiasD2(
                outputI = 2,
                outputJ = 3,
                optimizer = Sgd(Scheduler.Fix(rate = 0.01f)).d2(2, 3),
                weight = IOType.d2(2, 3) { i, j -> i * 2f + j },
            ),
            BiasD2(
                outputI = 2,
                outputJ = 3,
                optimizer = Sgd(Scheduler.Fix(rate = 0.01f)).d2(2, 3),
                weight = IOType.d2(2, 3) { i, j -> i * 2f + j },
            ),
        ),
        outputI = 2,
        outputJ = 3,
    )
    val input
        get() = Batch.of(
            IOType.d2(
                IOType.d1(3) { it * 2f },
                IOType.d1(3) { it * 3f },
            ),
            IOType.d2(
                IOType.d1(3) { it * 4f },
                IOType.d1(3) { it * 5f },
            ),
        )

    @Test
    fun `expect=スキップ接続を行う`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 6f, 12f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(4f, 12f, 20f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, 10f, 20f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(4f, 16f, 28f), actual = actual[1][1])
    }

    @Test
    fun `train=勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 12f, 24f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(8f, 24f, 40f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, 20f, 40f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(8f, 32f, 56f), actual = actual[1][1])
    }

    private fun builder() = NetworkBuilder.inputD2(
        converter = RawD2(2, 3),
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
