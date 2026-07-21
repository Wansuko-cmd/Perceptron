@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.GraphEnv
import kotlin.test.Test

class ReLUD3Test {
    val target
        get() = ReLUD3(inputI = 1, inputJ = 2, inputK = 2)
    val input
        get() = Batch.of(
            IOType.d3(1, 2, 2) { _, j, k -> if (j == 0) k.toFloat() else -k.toFloat() - 1f },
            IOType.d3(1, 2, 2) { _, j, k -> if (j == 0) k * 0.5f else -k * 1.5f - 2f },
        )

    @Test
    fun `expect=正の値のみ通し0以下は0にする`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 1f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(0f, 0.5f), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[1][0][1])
    }

    @Test
    fun `train=正の位置のみ勾配を通す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D3>

        fun grad(x: Float): Float = maxOf(x, 0f)

        assertContentEquals(expected = IOType.d1(grad(0f), grad(1f)), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(grad(-1f), grad(-2f)), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(grad(0f), grad(0.5f)), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(grad(-2f), grad(-3.5f)), actual = actual[1][0][1])
    }
}
