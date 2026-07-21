@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.scale.d2

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

class ScaleAxisD2Test {
    val target0
        get() = ScaleAxisD2(
            inputI = 2,
            inputJ = 2,
            axis = 0,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d1(2),
            weight = IOType.d1(2) { it.toFloat() },
        )

    val target1
        get() = ScaleAxisD2(
            inputI = 2,
            inputJ = 2,
            axis = 1,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d1(2),
            weight = IOType.d1(2) { it.toFloat() },
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
    fun `Axis0_expect=axis0で共通のスケール項`() = networkScopeTestRule {
        val actual = with(target0) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 3f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -1f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `Axis0_train=Axis0で共通の勾配を伝播`() = networkScopeTestRule {
        val actual = with(target0) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 3f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -1f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `Axis0_train=重みを更新する`() = networkScopeTestRule {
        val target = target0

        with(target) { _train(input = input, env = GraphEnv(), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 2.85f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -0.95f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `Axis1_expect=axis1で共通のスケール項`() = networkScopeTestRule {
        val actual = with(target1) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 3f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, -2f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -1f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `Axis1_train=Axis1で共通の勾配を伝播`() = networkScopeTestRule {
        val actual = with(target1) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 3f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, -2f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -1f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `Axis1_train=重みを更新する`() = networkScopeTestRule {
        val target = target1

        with(target) { _train(input = input, env = GraphEnv(), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 1.82f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 2.73f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, -1.82f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -0.91f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }
}
