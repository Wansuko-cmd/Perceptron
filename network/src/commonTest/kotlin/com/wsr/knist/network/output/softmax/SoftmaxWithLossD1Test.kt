@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.output.softmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.batchOf
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.get
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class SoftmaxWithLossD1Test {
    @Test
    fun `expect=softmaxを計算`() = networkTestRule {
        val target = SoftmaxWithLossD1(outputSize = 3, temperature = 1f)
        val input = batchOf(IOType.d1(1f, 2f, 3f))

        val actual = target._expect(input) as Batch<IOType.D1>

        assertEquals(expected = 0.0900f, actual = actual[0][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2447f, actual = actual[0][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.6652f, actual = actual[0][2].get(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=softmaxの逆伝播`() = networkTestRule {
        val target = SoftmaxWithLossD1(outputSize = 3, temperature = 1f)
        val input = batchOf(IOType.d1(1f, 2f, 3f))
        val label = batchOf(IOType.d1(1f, 3f, 5f))

        val actual = target._train(input = input, label = { label })
        val loss = actual.loss
        val delta = actual.delta as Batch<IOType.D1>

        assertEquals(expected = -1.4232f, actual = loss, absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.9099f, actual = delta[0][0].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -2.7552f, actual = delta[0][1].get(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -4.3347f, actual = delta[0][2].get(), absoluteTolerance = 1e-4f)
    }
}
