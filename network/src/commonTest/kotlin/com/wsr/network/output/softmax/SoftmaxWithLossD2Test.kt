@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.output.softmax

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import com.wsr.output.softmax.SoftmaxWithLossD2
import org.junit.Rule
import kotlin.test.Test
import kotlin.test.assertEquals

class SoftmaxWithLossD2Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    @Test
    fun `expect=softmaxを計算`() {
        val target = SoftmaxWithLossD2(outputX = 2, outputY = 2, temperature = 1f)
        val input = batchOf(IOType.d2(2, 2) { i, j -> i * 2f + j })

        val actual = target._expect(input) as Batch<IOType.D2>

        assertEquals(expected = 0.2689f, actual = actual[0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.7310f, actual = actual[0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2689f, actual = actual[0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.7310f, actual = actual[0][1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=softmaxの逆伝播`() {
        val target = SoftmaxWithLossD2(outputX = 2, outputY = 2, temperature = 1f)
        val input = batchOf(IOType.d2(2, 2) { i, j -> i * 2f + j })
        val label = batchOf(IOType.d2(2, 2) { i, j -> i * 4f + j * 2f })

        val actual = target._train(input = input, label = { label })
        val loss = actual.loss
        val delta = actual.delta as Batch<IOType.D2>

        assertEquals(expected = -1.0388f, actual = loss, absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.2689f, actual = delta[0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -1.2689f, actual = delta[0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = -3.7310f, actual = delta[0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -5.2689f, actual = delta[0][1][1], absoluteTolerance = 1e-4f)
    }
}
