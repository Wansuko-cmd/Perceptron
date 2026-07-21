@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.bias.d1

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import kotlin.test.Test

class BiasD1Test {
    val target
        get() = BiasD1(
            inputI = 3,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d1(3),
            weight = IOType.d1(3) { it.toFloat() },
        )

    val input get() = Batch.of(
        IOType.d1(3) { it * 2f },
        IOType.d1(3) { it * 3f },
    )

    @Test
    fun `expect=バイアス項`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d1(0f, 3f, 6f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(0f, 4f, 8f), actual = actual[1])
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d1(0f, 3f, 6f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(0f, 4f, 8f), actual = actual[1])
    }

    @Test
    fun `train=重みを更新する`() = networkScopeTestRule {
        val target = target

        with(target) { _train(input = input, env = GraphEnv(), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d1(0f, 2.965f, 5.93f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(0f, 3.965f, 7.93f), actual = actual[1])
    }
}
