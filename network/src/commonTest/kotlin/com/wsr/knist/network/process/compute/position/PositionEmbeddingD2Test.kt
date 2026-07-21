@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.position

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.GraphEnv
import kotlin.test.Test

class PositionEmbeddingD2Test {
    val target
        get() = PositionEmbeddingD2(
            inputI = 2,
            inputJ = 3,
            optimizer = Sgd(scheduler = Scheduler.Fix(rate = 0.01f)).d2(2, 3),
            weight = IOType.d2(2, 3) { i, j -> i * 2f + j },
        )
    val input
        get() = Batch.of(
            IOType.d2(
                IOType.d1(3) { it.toFloat() },
                IOType.d1(3) { it * 2f },
            ),
            IOType.d2(
                IOType.d1(3) { 10f % (it + 1) },
                IOType.d1(3) { it * 0.3f },
            ),
        )

    @Test
    fun `expect=学習型位置情報埋め込み`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 2f, 4f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(2f, 5f, 8f), actual = actual[0][1])

        assertContentEquals(expected = IOType.d1(0f, 1f, 3f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(2f, 3.3f, 4.6f), actual = actual[1][1])
    }

    @Test
    fun `train=勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 2f, 4f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(2f, 5f, 8f), actual = actual[0][1])

        assertContentEquals(expected = IOType.d1(0f, 1f, 3f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(2f, 3.3f, 4.6f), actual = actual[1][1])
    }

    @Test
    fun `train=重みを更新`() = networkScopeTestRule {
        val target = target

        with(target) { _train(input = input, env = GraphEnv(), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(0f, 1.985f, 3.9650f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(1.98f, 4.9585f, 7.9370f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0f, 0.985f, 2.9650f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(1.98f, 3.2584f, 4.537f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
