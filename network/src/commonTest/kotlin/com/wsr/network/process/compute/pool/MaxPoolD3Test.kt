@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.pool

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.d3
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import com.wsr.process.Context
import com.wsr.process.compute.pool.MaxPoolD3
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class MaxPoolD3Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target get() = MaxPoolD3(poolSize = 2, channel = 2, inputX = 2, inputY = 4)
    val input
        get() = batchOf(
            IOType.d3(
                IOType.d2(
                    IOType.d1(4) { it.toFloat() },
                    IOType.d1(4) { it * 2f },
                ),
                IOType.d2(
                    IOType.d1(4) { 10f % (it + 1) },
                    IOType.d1(4) { it * 0.3f },
                ),
            ),
            IOType.d3(
                IOType.d2(
                    IOType.d1(4) { it * it * 2f },
                    IOType.d1(4) { it + 5f },
                ),
                IOType.d2(
                    IOType.d1(4) { it % 1.5f },
                    IOType.d1(4) { 10f / (it - 5) },
                ),
            ),
        )

    @Test
    fun `expect=指定区間内での最大値を取得`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(2f, 4f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(0.3f, 1f), actual = actual[0][1][0])

        assertEquals(expected = IOType.d1(6f, 8f), actual = actual[1][0][0])
        assertEquals(expected = IOType.d1(1f, 1f), actual = actual[1][1][0])
    }

    @Test
    fun `train=勾配を伝播`() {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D3>

        assertEquals(expected = IOType.d1(0f, 0f, 0f, 0f), actual = actual[0][0][0])
        assertEquals(expected = IOType.d1(0f, 0f, 1f, 0f), actual = actual[0][1][0])

        assertEquals(expected = IOType.d1(0f, 0f, 8f, 0f), actual = actual[1][0][0])
        assertEquals(expected = IOType.d1(0f, 1f, 0f, 0f), actual = actual[1][1][0])
    }
}
