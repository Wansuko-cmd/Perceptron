@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.output.softmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.core.unwrap
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class SoftmaxWithLossD2Test {
    @Test
    fun `expect=softmaxを計算`() = networkScopeTestRule {
        val target = SoftmaxWithLossD2(outputX = 2, outputY = 2, temperature = 1f)
        val input = Batch.of(IOType.d2(2, 2) { i, j -> i * 2f + j })

        val actual = with(target) { _expect(input) } as Batch<IOType.D2>

        assertEquals(expected = 0.2689f, actual = actual[0][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.7310f, actual = actual[0][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2689f, actual = actual[0][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.7310f, actual = actual[0][1][1].unwrap(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=softmaxの逆伝播`() = networkScopeTestRule {
        val target = SoftmaxWithLossD2(outputX = 2, outputY = 2, temperature = 1f)
        val input = Batch.of(IOType.d2(2, 2) { i, j -> i * 2f + j })
        val label = Batch.of(IOType.d2(2, 2) { i, j -> i * 4f + j * 2f })

        val actual = with(target) { _train(input = input, label = { label }) }
        val loss = actual.loss.unwrap()
        val delta = actual.delta as Batch<IOType.D2>

        assertEquals(expected = -1.0388f, actual = loss, absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2689f, actual = delta[0][0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -1.2689f, actual = delta[0][0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -3.7310f, actual = delta[0][1][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -5.2689f, actual = delta[0][1][1].unwrap(), absoluteTolerance = 1e-4f)
    }
}
