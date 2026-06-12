@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.optimizer.momentum

import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.core.get
import com.wsr.knist.network.networkTestRule
import com.wsr.knist.network.optimizer.Scheduler
import kotlin.test.Test
import kotlin.test.assertEquals

class MomentumTest {
    @Test
    fun `D1_adapt=Momentumアルゴリズムで最適化`() = networkTestRule {
        val target = Momentum(scheduler = Scheduler.Fix(1f)).d1(3)
        val weight = IOType.d1(1f, 2f, 3f)
        val dw = IOType.d1(1f, 2f, 3f)

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual1[0].get())
        assertEquals(expected = 0f, actual = actual1[1].get())
        assertEquals(expected = 0f, actual = actual1[2].get())

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = -0.9000f, actual = actual2[0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -1.8000f, actual = actual2[1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -2.7000f, actual = actual2[2].get(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D2_adapt=Momentumアルゴリズムで最適化`() = networkTestRule {
        val target = Momentum(scheduler = Scheduler.Fix(1f)).d2(i = 2, j = 2)
        val weight = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val dw = IOType.d2(2, 2) { i, j -> i * 2f + j }

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual1[0][0].get())
        assertEquals(expected = 0f, actual = actual1[0][1].get())
        assertEquals(expected = 0f, actual = actual1[1][0].get())
        assertEquals(expected = 0f, actual = actual1[1][1].get())

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual2[0][0].get())
        assertEquals(expected = -0.9000f, actual = actual2[0][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -1.8000f, actual = actual2[1][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -2.7000f, actual = actual2[1][1].get(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D3_adapt=Momentumアルゴリズムで最適化`() = networkTestRule {
        val target = Momentum(scheduler = Scheduler.Fix(1f)).d3(i = 2, j = 2, k = 2)
        val weight = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k }
        val dw = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k }

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual1[0][0][0].get())
        assertEquals(expected = 0f, actual = actual1[0][0][1].get())
        assertEquals(expected = 0f, actual = actual1[0][1][0].get())
        assertEquals(expected = 0f, actual = actual1[0][1][1].get())
        assertEquals(expected = 0f, actual = actual1[1][0][0].get())
        assertEquals(expected = 0f, actual = actual1[1][0][1].get())
        assertEquals(expected = 0f, actual = actual1[1][1][0].get())
        assertEquals(expected = 0f, actual = actual1[1][1][1].get())

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual2[0][0][0].get())
        assertEquals(expected = -0.9000f, actual = actual2[0][0][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -1.8000f, actual = actual2[0][1][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -2.7000f, actual = actual2[0][1][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -3.6000f, actual = actual2[1][0][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -4.5000f, actual = actual2[1][0][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -5.4000f, actual = actual2[1][1][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -6.3000f, actual = actual2[1][1][1].get(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D4_adapt=Momentumアルゴリズムで最適化`() = networkTestRule {
        val target = Momentum(scheduler = Scheduler.Fix(1f)).d4(i = 2, j = 2, k = 2, l = 2)
        val weight = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }
        val dw = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual1[0][0][0][0].get())
        assertEquals(expected = 0f, actual = actual1[0][0][0][1].get())
        assertEquals(expected = 0f, actual = actual1[0][0][1][0].get())
        assertEquals(expected = 0f, actual = actual1[0][0][1][1].get())
        assertEquals(expected = 0f, actual = actual1[0][1][0][0].get())
        assertEquals(expected = 0f, actual = actual1[0][1][0][1].get())
        assertEquals(expected = 0f, actual = actual1[0][1][1][0].get())
        assertEquals(expected = 0f, actual = actual1[0][1][1][1].get())
        assertEquals(expected = 0f, actual = actual1[1][0][0][0].get())
        assertEquals(expected = 0f, actual = actual1[1][0][0][1].get())
        assertEquals(expected = 0f, actual = actual1[1][0][1][0].get())
        assertEquals(expected = 0f, actual = actual1[1][0][1][1].get())
        assertEquals(expected = 0f, actual = actual1[1][1][0][0].get())
        assertEquals(expected = 0f, actual = actual1[1][1][0][1].get())
        assertEquals(expected = 0f, actual = actual1[1][1][1][0].get())
        assertEquals(expected = 0f, actual = actual1[1][1][1][1].get())

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual2[0][0][0][0].get())
        assertEquals(expected = -0.9000f, actual = actual2[0][0][0][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -1.8000f, actual = actual2[0][0][1][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -2.7000f, actual = actual2[0][0][1][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -3.6000f, actual = actual2[0][1][0][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -4.5000f, actual = actual2[0][1][0][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -5.4000f, actual = actual2[0][1][1][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -6.3000f, actual = actual2[0][1][1][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -7.2000f, actual = actual2[1][0][0][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -8.1000f, actual = actual2[1][0][0][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -9.0000f, actual = actual2[1][0][1][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -9.9000f, actual = actual2[1][0][1][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -10.8000f, actual = actual2[1][1][0][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -11.7000f, actual = actual2[1][1][0][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -12.6000f, actual = actual2[1][1][1][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -13.5000f, actual = actual2[1][1][1][1].get(), absoluteTolerance = 1e-4f)
    }
}
