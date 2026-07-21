@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.debug

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test
import kotlin.test.assertNotNull

class DebugD1Test {
    val target
        get() = DebugD1(inputI = 3)
    val input
        get() = Batch.of(IOType.d1(1f, 2f, 3f))

    @Test
    fun `expect=入力をそのまま返しonInputを呼び出す`() = networkScopeTestRule {
        val target = target
        var captured: Batch<IOType.D1>? = null
        target.onInput = { captured = it }

        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D1>

        assertContentEquals(expected = input, actual = actual)
        assertContentEquals(expected = input, actual = assertNotNull(captured))
    }

    @Test
    fun `train=onInputとonDeltaを呼び出し、calcDeltaの結果をそのまま返す`() = networkScopeTestRule {
        val target = target
        var capturedInput: Batch<IOType.D1>? = null
        var capturedDelta: Batch<IOType.D1>? = null
        target.onInput = { capturedInput = it }
        target.onDelta = { capturedDelta = it }

        val delta = Batch.of(IOType.d1(10f, 20f, 30f))
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { delta })
        } as Batch<IOType.D1>

        assertContentEquals(expected = input, actual = assertNotNull(capturedInput))
        assertContentEquals(expected = delta, actual = assertNotNull(capturedDelta))
        assertContentEquals(expected = delta, actual = actual)
    }
}
