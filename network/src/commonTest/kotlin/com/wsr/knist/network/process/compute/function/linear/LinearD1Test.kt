@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.linear

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.Context
import kotlin.test.Test

class LinearD1Test {
    val target
        get() = LinearD1(inputI = 3)
    val input
        get() = Batch.of(IOType.d1(1f, 2f, 3f))

    @Test
    fun `expect=入力をそのまま返す`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D1>

        assertContentEquals(expected = input, actual = actual)
    }

    @Test
    fun `train=calcDeltaの結果をそのまま返す`() = networkScopeTestRule {
        val label = Batch.of(IOType.d1(10f, 20f, 30f))

        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { label })
        } as Batch<IOType.D1>

        assertContentEquals(expected = label, actual = actual)
    }
}
