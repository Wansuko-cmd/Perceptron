@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.dropout

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import com.wsr.network.assertEquals
import com.wsr.process.Context
import com.wsr.process.compute.dropout.DropoutD2
import org.junit.Rule
import kotlin.test.Test
import kotlin.test.assertEquals

class DropoutD2Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target get() = DropoutD2(outputX = 2, outputY = 2, ratio = 0.8f, seed = 0)
    val input
        get() = batchOf(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 3f + j * 2f },
        )

    @Test
    fun `expect=入力をそのまま返す`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(expected = input[0], actual = actual[0])
        assertEquals(expected = input[1], actual = actual[1])
    }

    @Test
    fun `train=dropoutを行いratioを掛け勾配を伝播`() {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(0f, 0f), actual = actual[0][0])
        assertEquals(expected = IOType.d1(3.125f, 4.6875f), actual = actual[0][1])
        assertEquals(expected = IOType.d1(0f, 0f), actual = actual[1][0])
        assertEquals(expected = IOType.d1(4.6875f, 7.8125f), actual = actual[1][1])
    }
}
