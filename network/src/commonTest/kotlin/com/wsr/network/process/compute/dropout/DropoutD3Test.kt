@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.dropout

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d3
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import com.wsr.process.Context
import com.wsr.process.compute.dropout.DropoutD3
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class DropoutD3Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target get() = DropoutD3(outputX = 2, outputY = 2, outputZ = 2, ratio = 0.8f, seed = 0)
    val input
        get() = batchOf(
            IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k },
            IOType.d3(2, 2, 2) { i, j, k -> i * 6f + j * 3f + k },
        )

    @Test
    fun `expect=入力をそのまま返す`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(expected = input[0], actual = actual[0])
        assertEquals(expected = input[1], actual = actual[1])
    }

    @Test
    fun `train=dropoutを行いratioを掛け勾配を伝播`() {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 0f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(3.125f, 4.6875f), actual = actual[0][0][1])
        assertEquals(expected = IOType.d1(6.25f, 7.8125f), actual = actual[0][1][0])
        assertEquals(expected = IOType.d1(9.375f, 10.9375f), actual = actual[0][1][1])
        assertEquals(expected = IOType.d1(0f, 0f), actual = actual[1][0][0])
        assertEquals(expected = IOType.d1(4.6875f, 6.25f), actual = actual[1][0][1])
        assertEquals(expected = IOType.d1(9.375f, 10.9375f), actual = actual[1][1][0])
        assertEquals(expected = IOType.d1(14.0625f, 15.625f), actual = actual[1][1][1])
    }
}
