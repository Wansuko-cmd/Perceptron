@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.optimizer

import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.core.get
import com.wsr.knist.core.unwrap
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class OptimizerTest {
    @Test
    fun `D1_adapt=勾配をmaxNormで正規化`() = networkScopeTestRule {
        val target = object : Optimizer.D1(_maxNorm = 1f) {
            override fun IOScope.adapt(weight: IOType.D1, dw: IOType.D1): IOType.D1 = dw
        }
        val weight = IOType.d1(0f, 0f, 0f)
        val dw = IOType.d1(1f, 2f, 3f)

        val actual = target.adapt(weight, dw)

        assertEquals(expected = 0.2672f, actual = actual[0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.5345f, actual = actual[1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.8017f, actual = actual[2].unwrap(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D2_adapt=勾配をmaxNormで正規化`() = networkScopeTestRule {
        val target = object : Optimizer.D2(_maxNorm = 1f) {
            override fun IOScope.adapt(weight: IOType.D2, dw: IOType.D2): IOType.D2 = dw
        }
        val weight = IOType.d2(2, 2) { _, _ -> 0f }
        val dw = IOType.d2(2, 2) { i, j -> i * 2f + j }

        val actual = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual[0][0].unwrap())
        assertEquals(expected = 0.2672f, actual = actual[0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.5345f, actual = actual[1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.8017f, actual = actual[1][1].unwrap(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D3_adapt=勾配をmaxNormで正規化`() = networkScopeTestRule {
        val target = object : Optimizer.D3(_maxNorm = 1f) {
            override fun IOScope.adapt(weight: IOType.D3, dw: IOType.D3): IOType.D3 = dw
        }
        val weight = IOType.d3(2, 2, 2) { _, _, _ -> 0f }
        val dw = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k }

        val actual = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual[0][0][0].unwrap())
        assertEquals(expected = 0.0845f, actual = actual[0][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.1690f, actual = actual[0][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2535f, actual = actual[0][1][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3380f, actual = actual[1][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.4225f, actual = actual[1][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.5070f, actual = actual[1][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.5916f, actual = actual[1][1][1].unwrap(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D4_adapt=勾配をmaxNormで正規化`() = networkScopeTestRule {
        val target = object : Optimizer.D4(_maxNorm = 1f) {
            override fun IOScope.adapt(weight: IOType.D4, dw: IOType.D4): IOType.D4 = dw
        }
        val weight = IOType.d4(2, 2, 2, 2) { _, _, _, _ -> 0f }
        val dw = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }

        val actual = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual[0][0][0][0].unwrap())
        assertEquals(expected = 0.0283f, actual = actual[0][0][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.0567f, actual = actual[0][0][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.0851f, actual = actual[0][0][1][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.1135f, actual = actual[0][1][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.1419f, actual = actual[0][1][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.1703f, actual = actual[0][1][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.1987f, actual = actual[0][1][1][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2271f, actual = actual[1][0][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2555f, actual = actual[1][0][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2839f, actual = actual[1][0][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3123f, actual = actual[1][0][1][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3407f, actual = actual[1][1][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3691f, actual = actual[1][1][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.3975f, actual = actual[1][1][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.4259f, actual = actual[1][1][1][1].unwrap(), absoluteTolerance = 1e-4f)
    }
}
