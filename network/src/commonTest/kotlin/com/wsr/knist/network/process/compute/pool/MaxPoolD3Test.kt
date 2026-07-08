@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.pool

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.Context
import kotlin.test.Test

class MaxPoolD3Test {
    val target get() = MaxPoolD3(poolSize = 2, channel = 2, height = 2, width = 4, padding = 0)
    val input
        get() = Batch.of(
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
    fun `expect=指定区間内での最大値を取得`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(2f, 6f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(0.3f, 2f), actual = actual[0][1][0])

        assertContentEquals(expected = IOType.d1(6f, 18f), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(1f, 0.5f), actual = actual[1][1][0])
    }

    @Test
    fun `train=勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 0f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 2f), actual = actual[0][1][0])

        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 18f), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(0f, 1f, 0.5f, 0f), actual = actual[1][1][0])
    }
}
