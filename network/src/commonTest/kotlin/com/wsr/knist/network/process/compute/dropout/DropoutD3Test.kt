@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.dropout

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class DropoutD3Test {
    val target get() = DropoutD3(inputI = 2, inputJ = 2, inputK = 2, ratio = 0.8f, seed = 0)
    val input
        get() = Batch.of(
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            IOType.d3(2, 2, 2) { i, j, k -> i * 6f + j * 3f + k },
        )

    @Test
    fun `expect=入力をそのまま返す`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(expected = input[0], actual = actual[0])
        assertContentEquals(expected = input[1], actual = actual[1])
    }

    @Test
    fun `train=dropoutを行いratioを掛け勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(3.125f, 4.6875f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(6.25f, 7.8125f), actual = actual[0][1][0])
        assertContentEquals(expected = IOType.d1(9.375f, 10.9375f), actual = actual[0][1][1])
        assertContentEquals(expected = IOType.d1(0f, 1.5625f), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(0f, 6.25f), actual = actual[1][0][1])
        assertContentEquals(expected = IOType.d1(9.375f, 10.9375f), actual = actual[1][1][0])
        assertContentEquals(expected = IOType.d1(14.0625f, 15.625f), actual = actual[1][1][1])
    }
}
