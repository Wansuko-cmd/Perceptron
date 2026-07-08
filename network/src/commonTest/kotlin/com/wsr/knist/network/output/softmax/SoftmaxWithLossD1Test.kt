@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.output.softmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.get
import com.wsr.knist.core.unwrap
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class SoftmaxWithLossD1Test {
    @Test
    fun `expect=softmaxを計算`() = networkScopeTestRule {
        val target = SoftmaxWithLossD1(temperature = 0.8f)
        val input = Batch.of(IOType.d1(1f, 2f, 3f))

        val actual = with(target) { _expect(input) } as Batch<IOType.D1>

        assertEquals(expected = 0.0600f, actual = actual[0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2093f, actual = actual[0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.7307f, actual = actual[0][2].unwrap(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=softmaxの逆伝播`() = networkScopeTestRule {
        val target = SoftmaxWithLossD1(temperature = 0.8f)
        val input = Batch.of(IOType.d1(1f, 2f, 3f))
        val label = Batch.of(IOType.d1(1f, 3f, 5f))

        val actual = with(target) { _train(input = input, label = { label }) }
        val loss = actual.loss.unwrap()
        val delta = actual.delta as Batch<IOType.D1>

        assertEquals(expected = -1.4682f, actual = loss, absoluteTolerance = 1e-4f)
        assertEquals(expected = -1.1750f, actual = delta[0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -3.4883f, actual = delta[0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -5.3367f, actual = delta[0][2].unwrap(), absoluteTolerance = 1e-4f)
    }
}
