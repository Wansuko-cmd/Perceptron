@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.dropout

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.GraphEnv
import kotlin.test.Test

class DropoutD1Test {
    val target get() = DropoutD1(inputI = 3, ratio = 0.8f, seed = 0)
    val input
        get() = Batch.of(
            IOType.d1(3) { it * 2f },
            IOType.d1(3) { it * 3f },
        )

    @Test
    fun `expect=入力をそのまま返す`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D1>

        assertContentEquals(expected = input[0], actual = actual[0])
        assertContentEquals(expected = input[1], actual = actual[1])
    }

    @Test
    fun `train=dropoutを行いratioを掛け勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d1(0f, 0f, 6.25f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(0f, 4.6875f, 9.375f), actual = actual[1])
    }
}
