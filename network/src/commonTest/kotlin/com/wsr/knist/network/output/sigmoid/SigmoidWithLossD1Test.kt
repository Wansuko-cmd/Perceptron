@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.output.sigmoid

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.batchOf
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.get
import com.wsr.knist.core.unwrap
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class SigmoidWithLossD1Test {
    @Test
    fun `expect=sigmoidを計算`() = networkTestRule {
        val target = SigmoidWithLossD1(outputSize = 3)
        val input = batchOf(IOType.d1(1f, 2f, 3f))

        val actual = target._expect(input) as Batch<IOType.D1>

        assertEquals(expected = 0.731f, actual = actual[0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.8807f, actual = actual[0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9525f, actual = actual[0][2].unwrap(), absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=sigmoidの逆伝播`() = networkTestRule {
        val target = SigmoidWithLossD1(outputSize = 3)
        val input = batchOf(IOType.d1(1f, 2f, 3f))
        val label = batchOf(IOType.d1(1f, 3f, 5f))

        val actual = target._train(input = input, label = { label })
        val loss = actual.loss
        val delta = actual.delta as Batch<IOType.D1>

        assertEquals(expected = -15.5112f, actual = loss, absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.2689f, actual = delta[0][0].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -2.1192f, actual = delta[0][1].unwrap(), absoluteTolerance = 1e-4f)
        assertEquals(expected = -4.0474f, actual = delta[0][2].unwrap(), absoluteTolerance = 1e-4f)
    }
}
