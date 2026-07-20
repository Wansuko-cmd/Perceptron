@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.Context
import kotlin.test.Test

class ReLUD2Test {
    val target
        get() = ReLUD2(inputI = 2, inputJ = 2)
    val input
        get() = Batch.of(
            IOType.d2(2, 2) { i, j -> if (i == 0) j.toFloat() else -j.toFloat() - 1f },
            IOType.d2(2, 2) { i, j -> if (i == 0) j * 0.5f else -j * 1.5f - 2f },
        )

    @Test
    fun `expect=正の値のみ通し0以下は0にする`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 1f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, 0.5f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[1][1])
    }

    @Test
    fun `train=正の位置のみ勾配を通す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D2>

        fun grad(x: Float): Float = maxOf(x, 0f)

        assertContentEquals(expected = IOType.d1(grad(0f), grad(1f)), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(grad(-1f), grad(-2f)), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(grad(0f), grad(0.5f)), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(grad(-2f), grad(-3.5f)), actual = actual[1][1])
    }
}
