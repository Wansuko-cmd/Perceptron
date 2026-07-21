@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.tanh

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.GraphEnv
import kotlin.math.tanh
import kotlin.test.Test

class TanhD1Test {
    val target
        get() = TanhD1(inputI = 4)
    val input
        get() = Batch.of(
            IOType.d1(0f, 1f, -1f, 2f),
            IOType.d1(0.5f, -0.5f, 1.5f, -2f),
        )

    @Test
    fun `expect=双曲線正接を適用する`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d0(tanh(0f)), actual = actual[0][0])
        assertContentEquals(expected = IOType.d0(tanh(1f)), actual = actual[0][1])
        assertContentEquals(expected = IOType.d0(tanh(-1f)), actual = actual[0][2])
        assertContentEquals(expected = IOType.d0(tanh(2f)), actual = actual[0][3])
    }

    @Test
    fun `train=1-tanh^2を勾配に掛けて返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D1>

        fun grad(x: Float): Float {
            val t = tanh(x)
            return t * (1f - t * t)
        }

        assertContentEquals(expected = IOType.d0(grad(0.5f)), actual = actual[1][0])
        assertContentEquals(expected = IOType.d0(grad(-0.5f)), actual = actual[1][1])
        assertContentEquals(expected = IOType.d0(grad(1.5f)), actual = actual[1][2])
        assertContentEquals(expected = IOType.d0(grad(-2f)), actual = actual[1][3])
    }
}
