@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.scale.d2

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import kotlin.test.Test

class ScaleD2Test {
    val target
        get() = ScaleD2(
            inputI = 2,
            inputJ = 2,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d2(2, 2),
            weight = IOType.d2(2, 2) { i, j -> i * 2f + j },
        )

    val input
        get() = Batch.of(
            IOType.d2(
                IOType.d1(2) { it * 2f },
                IOType.d1(2) { it * 3f },
            ),
            IOType.d2(
                IOType.d1(2) { it * -2f },
                IOType.d1(2) { it * -1f },
            ),
        )

    @Test
    fun `expect=スケール項`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 9f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, -2f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -3f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 27f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, -2f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -9f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=重みを更新する`() = networkScopeTestRule {
        val target = target

        with(target) { _train(input = input, env = GraphEnv(), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 1.92f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 8.5499f), actual = actual[0][1], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -1.92f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -2.85f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }
}
