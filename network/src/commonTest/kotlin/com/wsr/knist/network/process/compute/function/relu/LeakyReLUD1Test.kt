@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.GraphEnv
import kotlin.test.Test

class LeakyReLUD1Test {
    val target
        get() = LeakyReLUD1(inputI = 4)
    val input
        get() = Batch.of(
            IOType.d1(0f, 1f, -1f, 2f),
            IOType.d1(0.5f, -0.5f, 1.5f, -2f),
        )

    fun leaky(x: Float): Float = if (x > 0f) x else 0.01f * x

    @Test
    fun `expect=正はそのまま、0以下は0_01倍にする`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d0(leaky(0f)), actual = actual[0][0])
        assertContentEquals(expected = IOType.d0(leaky(1f)), actual = actual[0][1])
        assertContentEquals(expected = IOType.d0(leaky(-1f)), actual = actual[0][2])
        assertContentEquals(expected = IOType.d0(leaky(2f)), actual = actual[0][3])
    }

    @Test
    fun `train=正の位置はそのまま、0以下は0_01倍の勾配を通す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D1>

        fun grad(x: Float): Float = if (x > 0f) leaky(x) else 0.01f * leaky(x)

        assertContentEquals(expected = IOType.d0(grad(0.5f)), actual = actual[1][0])
        assertContentEquals(expected = IOType.d0(grad(-0.5f)), actual = actual[1][1])
        assertContentEquals(expected = IOType.d0(grad(1.5f)), actual = actual[1][2])
        assertContentEquals(expected = IOType.d0(grad(-2f)), actual = actual[1][3])
    }
}
