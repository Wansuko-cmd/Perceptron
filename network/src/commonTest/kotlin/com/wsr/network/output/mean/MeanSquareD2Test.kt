@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.output.mean

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class MeanSquareD2Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    @Test
    fun `expect=そのまま返す`() {
        val target = MeanSquareD2(outputX = 2, outputY = 2)
        val input = batchOf(IOType.d2(2, 2) { i, j -> i * 2f + j })

        val actual = target._expect(input)

        assertEquals(expected = input, actual = actual)
    }

    @Test
    fun `train=二乗平均誤差`() {
        val target = MeanSquareD2(outputX = 2, outputY = 2)
        val input = batchOf(IOType.d2(2, 2) { i, j -> i * 2f + j })
        val label = batchOf(IOType.d2(2, 2) { i, j -> i * 4f + j * 2f })

        val actual = target._train(input = input, label = { label })
        val loss = actual.loss
        val delta = actual.delta as Batch<IOType.D2>

        assertEquals(expected = 1.75f, actual = loss)
        assertEquals(expected = 0f, actual = delta[0][0][0])
        assertEquals(expected = -1f, actual = delta[0][0][1])
        assertEquals(expected = -2f, actual = delta[0][1][0])
        assertEquals(expected = -3f, actual = delta[0][1][1])
    }
}
