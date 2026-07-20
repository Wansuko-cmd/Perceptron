@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.debug

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.Context
import kotlin.test.Test
import kotlin.test.assertNotNull

class DebugD3Test {
    val target
        get() = DebugD3(inputI = 1, inputJ = 2, inputK = 2)
    val input
        get() = Batch.of(IOType.d3(1, 2, 2) { _, j, k -> j * 2f + k })

    @Test
    fun `expect=入力をそのまま返しonInputを呼び出す`() = networkScopeTestRule {
        val target = target
        var captured: Batch<IOType.D3>? = null
        target.onInput = { captured = it }

        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D3>

        assertContentEquals(expected = input, actual = actual)
        assertContentEquals(expected = input, actual = assertNotNull(captured))
    }

    @Test
    fun `train=onInputとonDeltaを呼び出し、calcDeltaの結果をそのまま返す`() = networkScopeTestRule {
        val target = target
        var capturedInput: Batch<IOType.D3>? = null
        var capturedDelta: Batch<IOType.D3>? = null
        target.onInput = { capturedInput = it }
        target.onDelta = { capturedDelta = it }

        val delta = Batch.of(IOType.d3(1, 2, 2) { _, j, k -> j * 10f + k })
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { delta })
        } as Batch<IOType.D3>

        assertContentEquals(expected = input, actual = assertNotNull(capturedInput))
        assertContentEquals(expected = delta, actual = assertNotNull(capturedDelta))
        assertContentEquals(expected = delta, actual = actual)
    }
}
