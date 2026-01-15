@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.output.sigmoid

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class SigmoidWithLossD2Test {
    @Test
    fun `expect=sigmoidを計算`() = networkTestRule {
        val target = SigmoidWithLossD2(outputX = 2, outputY = 2)
        val input = batchOf(IOType.d2(2, 2) { i, j -> i * 2f + j })

        val actual = target._expect(input) as Batch<IOType.D2>

        assertEquals(expected = 0.5f, actual = actual[0][0][0])
        assertEquals(expected = 0.7310f, actual = actual[0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.8807f, actual = actual[0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9525f, actual = actual[0][1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=sigmoidの逆伝播`() = networkTestRule {
        val target = SigmoidWithLossD2(outputX = 2, outputY = 2)
        val input = batchOf(IOType.d2(2, 2) { i, j -> i * 2f + j })
        val label = batchOf(IOType.d2(2, 2) { i, j -> i * 4f + j * 2f })

        val actual = target._train(input = input, label = { label })
        val loss = actual.loss
        val delta = actual.delta as Batch<IOType.D2>

        assertEquals(expected = -20.818f, actual = loss, absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.5f, actual = delta[0][0][0])
        assertEquals(expected = -1.2689f, actual = delta[0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -3.1192f, actual = delta[0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -5.0474f, actual = delta[0][1][1], absoluteTolerance = 1e-4f)
    }
}
