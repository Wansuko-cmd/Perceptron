@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.dropout

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.network.NetworkTestRule
import com.wsr.process.Context
import com.wsr.process.compute.dropout.DropoutD1
import org.junit.Rule
import kotlin.test.Test
import kotlin.test.assertEquals

class DropoutD1Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target get() = DropoutD1(outputSize = 3, ratio = 0.8f, seed = 0)
    val input
        get() = batchOf(
            IOType.d1(3) { it * 2f },
            IOType.d1(3) { it * 3f },
        )

    @Test
    fun `expect=入力をそのまま返す`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D1>

        assertEquals(expected = input[0], actual = actual[0])
        assertEquals(expected = input[1], actual = actual[1])
    }

    @Test
    fun `train=dropoutを行いratioを掛け勾配を伝播`() {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D1>

        assertEquals(expected = IOType.d1(0f, 0f, 6.25f), actual = actual[0])
        assertEquals(expected = IOType.d1(0f, 0f, 9.375f), actual = actual[1])
    }
}
