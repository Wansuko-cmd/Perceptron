@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.tanh

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.math.tanh
import kotlin.test.Test

class TanhD2Test {
    val target
        get() = TanhD2(inputI = 2, inputJ = 2)
    val input
        get() = Batch.of(
            IOType.d2(2, 2) { i, j -> if (i == 0) j.toFloat() else -j.toFloat() - 1f },
            IOType.d2(2, 2) { i, j -> if (i == 0) j * 0.5f else -j * 1.5f - 2f },
        )

    @Test
    fun `expect=双曲線正接を適用する`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(tanh(0f), tanh(1f)), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(tanh(-1f), tanh(-2f)), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(tanh(0f), tanh(0.5f)), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(tanh(-2f), tanh(-3.5f)), actual = actual[1][1])
    }

    @Test
    fun `train=1-tanh^2を勾配に掛けて返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        fun grad(x: Float): Float {
            val t = tanh(x)
            return t * (1f - t * t)
        }

        assertContentEquals(expected = IOType.d1(grad(0f), grad(1f)), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(grad(-1f), grad(-2f)), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(grad(0f), grad(0.5f)), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(grad(-2f), grad(-3.5f)), actual = actual[1][1])
    }
}
