@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.optimizer.adam

import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.core.get
import com.wsr.knist.core.unwrap
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import kotlin.test.Test
import kotlin.test.assertEquals

class AdamTest {
    @Test
    fun `D1_adapt=Adamアルゴリズムで最適化`() = networkScopeTestRule {
        val target = Adam(scheduler = Scheduler.Fix(1f)).d1(3)
        val weight = IOType.d1(1f, 2f, 3f)
        val dw = IOType.d1(1f, 2f, 3f)

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0.000f, actual = actual1[0].unwrap())
        assertEquals(expected = 1.000f, actual = actual1[1].unwrap())
        assertEquals(expected = 2.000f, actual = actual1[2].unwrap())

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = 0.000f, actual = actual2[0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.000f, actual = actual2[1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 2.000f, actual = actual2[2].unwrap(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D2_adapt=Adamアルゴリズムで最適化`() = networkScopeTestRule {
        val target = Adam(scheduler = Scheduler.Fix(1f)).d2(i = 2, j = 2)
        val weight = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val dw = IOType.d2(2, 2) { i, j -> i * 2f + j }

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual1[0][0].unwrap())
        assertEquals(expected = 0f, actual = actual1[0][1].unwrap())
        assertEquals(expected = 1f, actual = actual1[1][0].unwrap())
        assertEquals(expected = 2f, actual = actual1[1][1].unwrap())

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual2[0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0f, actual = actual2[0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 1f, actual = actual2[1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 2f, actual = actual2[1][1].unwrap(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D3_adapt=Adamアルゴリズムで最適化`() = networkScopeTestRule {
        val target = Adam(scheduler = Scheduler.Fix(1f)).d3(i = 2, j = 2, k = 2)
        val weight = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k }
        val dw = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k }

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual1[0][0][0].unwrap())
        assertEquals(expected = 0f, actual = actual1[0][0][1].unwrap())
        assertEquals(expected = 1f, actual = actual1[0][1][0].unwrap())
        assertEquals(expected = 2f, actual = actual1[0][1][1].unwrap())
        assertEquals(expected = 3f, actual = actual1[1][0][0].unwrap())
        assertEquals(expected = 4f, actual = actual1[1][0][1].unwrap())
        assertEquals(expected = 5f, actual = actual1[1][1][0].unwrap())
        assertEquals(expected = 6f, actual = actual1[1][1][1].unwrap())

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = 0.0000f, actual = actual2[0][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.0000f, actual = actual2[0][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.0000f, actual = actual2[0][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 2.0000f, actual = actual2[0][1][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 3.0000f, actual = actual2[1][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 4.0000f, actual = actual2[1][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 5.0000f, actual = actual2[1][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 6.0000f, actual = actual2[1][1][1].unwrap(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D4_adapt=Adamアルゴリズムで最適化`() = networkScopeTestRule {
        val target = Adam(scheduler = Scheduler.Fix(1f)).d4(i = 2, j = 2, k = 2, l = 2)
        val weight = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }
        val dw = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual1[0][0][0][0].unwrap())
        assertEquals(expected = 0f, actual = actual1[0][0][0][1].unwrap())
        assertEquals(expected = 1f, actual = actual1[0][0][1][0].unwrap())
        assertEquals(expected = 2f, actual = actual1[0][0][1][1].unwrap())
        assertEquals(expected = 3f, actual = actual1[0][1][0][0].unwrap())
        assertEquals(expected = 4f, actual = actual1[0][1][0][1].unwrap())
        assertEquals(expected = 5f, actual = actual1[0][1][1][0].unwrap())
        assertEquals(expected = 6f, actual = actual1[0][1][1][1].unwrap())
        assertEquals(expected = 7f, actual = actual1[1][0][0][0].unwrap())
        assertEquals(expected = 8f, actual = actual1[1][0][0][1].unwrap())
        assertEquals(expected = 9f, actual = actual1[1][0][1][0].unwrap())
        assertEquals(expected = 10f, actual = actual1[1][0][1][1].unwrap())
        assertEquals(expected = 11f, actual = actual1[1][1][0][0].unwrap())
        assertEquals(expected = 12f, actual = actual1[1][1][0][1].unwrap())
        assertEquals(expected = 13f, actual = actual1[1][1][1][0].unwrap())
        assertEquals(expected = 14f, actual = actual1[1][1][1][1].unwrap())

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = 0.0000f, actual = actual2[0][0][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.0000f, actual = actual2[0][0][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.0000f, actual = actual2[0][0][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 2.0000f, actual = actual2[0][0][1][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 3.0000f, actual = actual2[0][1][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 4.0000f, actual = actual2[0][1][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 5.0000f, actual = actual2[0][1][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 6.0000f, actual = actual2[0][1][1][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 7.0000f, actual = actual2[1][0][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 8.0000f, actual = actual2[1][0][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 9.0000f, actual = actual2[1][0][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 10.0000f, actual = actual2[1][0][1][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 11.0000f, actual = actual2[1][1][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 12.0000f, actual = actual2[1][1][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 13.0000f, actual = actual2[1][1][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 14.0000f, actual = actual2[1][1][1][1].unwrap(), absoluteTolerance = 1e-4f)
    }
}
