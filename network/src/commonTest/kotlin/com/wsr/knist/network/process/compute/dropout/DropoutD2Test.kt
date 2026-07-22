@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.dropout

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class DropoutD2Test {
    val target get() = DropoutD2(inputI = 2, inputJ = 2, ratio = 0.8f, seed = 0)
    val input
        get() = Batch.of(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 3f + j * 2f },
        )

    @Test
    fun `expect=入力をそのまま返す`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = input[0], actual = actual[0])
        assertContentEquals(expected = input[1], actual = actual[1])
    }

    @Test
    fun `train=dropoutを行いratioを掛け勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 0f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(3.125f, 4.6875f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, 3.125f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(4.6875f, 7.8125f), actual = actual[1][1])
    }
}
