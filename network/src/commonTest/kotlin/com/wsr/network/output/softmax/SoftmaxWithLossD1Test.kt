@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.output.softmax

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class SoftmaxWithLossD1Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    @Test
    fun `expect=softmaxを計算`() {
        val target = SoftmaxWithLossD1(outputSize = 3, temperature = 1f)
        val input = batchOf(IOType.d1(1f, 2f, 3f))

        val actual = target._expect(input) as Batch<IOType.D1>

        assertEquals(expected = 0.0900f, actual = actual[0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2447f, actual = actual[0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.6652f, actual = actual[0][2], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=softmaxの逆伝播`() {
        val target = SoftmaxWithLossD1(outputSize = 3, temperature = 1f)
        val input = batchOf(IOType.d1(1f, 2f, 3f))
        val label = batchOf(IOType.d1(1f, 3f, 5f))

        val actual = target._train(input = input, label = { label })
        val loss = actual.loss
        val delta = actual.delta as Batch<IOType.D1>

        assertEquals(expected = -1.4232f, actual = loss, absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.9099f, actual = delta[0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -2.7552f, actual = delta[0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -4.3347f, actual = delta[0][2], absoluteTolerance = 1e-4f)
    }
}
