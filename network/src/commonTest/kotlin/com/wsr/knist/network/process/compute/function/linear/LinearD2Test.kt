@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.linear

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.GraphEnv
import kotlin.test.Test

class LinearD2Test {
    val target
        get() = LinearD2(inputI = 2, inputJ = 2)
    val input
        get() = Batch.of(IOType.d2(2, 2) { i, j -> i * 2f + j })

    @Test
    fun `expect=入力をそのまま返す`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = input, actual = actual)
    }

    @Test
    fun `train=calcDeltaの結果をそのまま返す`() = networkScopeTestRule {
        val label = Batch.of(IOType.d2(2, 2) { i, j -> i * 10f + j })

        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { label })
        } as Batch<IOType.D2>

        assertContentEquals(expected = label, actual = actual)
    }
}
