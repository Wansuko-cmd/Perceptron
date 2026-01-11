@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.pool

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import com.wsr.network.process.Context
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class MaxPoolD2Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target get() = MaxPoolD2(poolSize = 2, channel = 2, inputSize = 4)
    val input
        get() = batchOf(
            IOType.d2(
                IOType.d1(4) { it.toFloat() },
                IOType.d1(4) { it * 2f },
            ),
            IOType.d2(
                IOType.d1(4) { 10f % (it + 1) },
                IOType.d1(4) { it * 0.3f },
            ),
        )

    @Test
    fun `expect=指定区間内での最大値を取得`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(1f, 2f), actual = actual[0][0])
        assertEquals(expected = IOType.d1(2f, 4f), actual = actual[0][1])

        assertEquals(expected = IOType.d1(0f, 1f), actual = actual[1][0])
        assertEquals(expected = IOType.d1(0.3f, 0.6f), actual = actual[1][1])
    }

    @Test
    fun `train=勾配を伝播`() {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(0f, 1f, 2f, 0f), actual = actual[0][0])
        assertEquals(expected = IOType.d1(0f, 2f, 4f, 0f), actual = actual[0][1])

        assertEquals(expected = IOType.d1(0f, 0f, 1f, 0f), actual = actual[1][0])
        assertEquals(expected = IOType.d1(0f, 0.3f, 0.6f, 0f), actual = actual[1][1])
    }
}
