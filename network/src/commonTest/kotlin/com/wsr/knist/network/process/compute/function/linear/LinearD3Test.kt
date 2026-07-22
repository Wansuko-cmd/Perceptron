@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.linear

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class LinearD3Test {
    val target
        get() = LinearD3(inputI = 1, inputJ = 2, inputK = 2)
    val input
        get() = Batch.of(IOType.d3(1, 2, 2) { _, j, k -> j * 2f + k })

    @Test
    fun `expect=入力をそのまま返す`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(expected = input, actual = actual)
    }

    @Test
    fun `train=calcDeltaの結果をそのまま返す`() = networkScopeTestRule {
        val label = Batch.of(IOType.d3(1, 2, 2) { _, j, k -> j * 10f + k })

        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { label })
        } as Batch<IOType.D3>

        assertContentEquals(expected = label, actual = actual)
    }
}
