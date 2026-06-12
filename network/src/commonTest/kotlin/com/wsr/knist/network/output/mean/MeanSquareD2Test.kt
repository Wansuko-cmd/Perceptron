@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.output.mean

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.batchOf
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.core.unwrap
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class MeanSquareD2Test {
    @Test
    fun `expect=そのまま返す`() = networkTestRule {
        val target = MeanSquareD2(outputX = 2, outputY = 2)
        val input = batchOf(IOType.d2(2, 2) { i, j -> i * 2f + j })

        val actual = target._expect(input)

        assertEquals(expected = input, actual = actual)
    }

    @Test
    fun `train=二乗平均誤差`() = networkTestRule {
        val target = MeanSquareD2(outputX = 2, outputY = 2)
        val input = batchOf(IOType.d2(2, 2) { i, j -> i * 2f + j })
        val label = batchOf(IOType.d2(2, 2) { i, j -> i * 4f + j * 2f })

        val actual = target._train(input = input, label = { label })
        val loss = actual.loss.unwrap()
        val delta = actual.delta as Batch<IOType.D2>

        assertEquals(expected = 1.75f, actual = loss)
        assertEquals(expected = 0f, actual = delta[0][0][0].unwrap())
        assertEquals(expected = -1f, actual = delta[0][0][1].unwrap())
        assertEquals(expected = -2f, actual = delta[0][1][0].unwrap())
        assertEquals(expected = -3f, actual = delta[0][1][1].unwrap())
    }
}
