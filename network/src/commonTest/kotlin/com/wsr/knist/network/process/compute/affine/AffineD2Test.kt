@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.affine

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

class AffineD2Test {
    val target
        get() = AffineD2(
            channel = 2,
            inputJ = 3,
            outputJ = 4,
            optimizer = Sgd(scheduler = Scheduler.Fix(rate = 0.01f)).d2(3, 4),
            weight = IOType.d2(3, 4) { i, j -> i * 2f + j },
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
    fun `expect=全結合`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(20f, 26f, 32f, 38f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(30f, 39f, 48f, 57f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(40f, 52f, 64f, 76f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(50f, 65f, 80f, 95f), actual = actual[1][1])
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(204f, 436f, 668f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(306f, 654f, 1002f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(408f, 872f, 1336f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(510f, 1090f, 1670f), actual = actual[1][1])
    }

    @Test
    fun `train=重みを更新する`() = networkScopeTestRule {
        val target = target

        with(target) { _train(input = input, env = GraphEnv(), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(-7.0000f, -9.1f, -11.2000f, -13.3000f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-10.5000f, -13.6500f, -16.8000f, -19.95f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-14.0000f, -18.2f, -22.3999f, -26.6000f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-17.5000f, -22.7500f, -27.9999f, -33.2500f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
