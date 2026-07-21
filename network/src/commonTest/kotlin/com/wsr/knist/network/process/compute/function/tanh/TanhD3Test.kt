@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.tanh

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.GraphEnv
import kotlin.math.tanh
import kotlin.test.Test

class TanhD3Test {
    val target
        get() = TanhD3(inputI = 1, inputJ = 2, inputK = 2)
    val input
        get() = Batch.of(
            IOType.d3(1, 2, 2) { _, j, k -> if (j == 0) k.toFloat() else -k.toFloat() - 1f },
            IOType.d3(1, 2, 2) { _, j, k -> if (j == 0) k * 0.5f else -k * 1.5f - 2f },
        )

    @Test
    fun `expect=双曲線正接を適用する`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(tanh(0f), tanh(1f)), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(tanh(-1f), tanh(-2f)), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(tanh(0f), tanh(0.5f)), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(tanh(-2f), tanh(-3.5f)), actual = actual[1][0][1])
    }

    @Test
    fun `train=1-tanh^2を勾配に掛けて返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D3>

        fun grad(x: Float): Float {
            val t = tanh(x)
            return t * (1f - t * t)
        }

        assertContentEquals(expected = IOType.d1(grad(0f), grad(1f)), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(grad(-1f), grad(-2f)), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(grad(0f), grad(0.5f)), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(grad(-2f), grad(-3.5f)), actual = actual[1][0][1])
    }
}
