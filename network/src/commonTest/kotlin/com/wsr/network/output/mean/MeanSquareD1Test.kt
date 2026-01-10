@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.output.mean

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import com.wsr.output.mean.MeanSquareD1
import org.junit.Rule
import kotlin.test.Test
import kotlin.test.assertEquals

class MeanSquareD1Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    @Test
    fun `expect=そのまま返す`() {
        val target = MeanSquareD1(outputSize = 3)
        val input = batchOf(IOType.d1(1f, 2f, 3f))

        val actual = target._expect(input)

        assertEquals(expected = input, actual = actual)
    }

    @Test
    fun `train=二乗平均誤差`() {
        val target = MeanSquareD1(outputSize = 3)
        val input = batchOf(IOType.d1(1f, 2f, 3f))
        val label = batchOf(IOType.d1(1f, 3f, 5f))

        val actual = target._train(input = input, label = { label })
        val loss = actual.loss
        val delta = actual.delta as Batch<IOType.D1>

        assertEquals(expected = 0.8333f, actual = loss, absoluteTolerance = 1e-4f)
        assertEquals(expected = 0f, actual = delta[0][0])
        assertEquals(expected = -1f, actual = delta[0][1])
        assertEquals(expected = -2f, actual = delta[0][2])
    }
}
