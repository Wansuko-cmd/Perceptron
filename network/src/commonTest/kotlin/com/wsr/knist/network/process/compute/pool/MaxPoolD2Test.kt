@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.pool

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.Context
import kotlin.test.Test

class MaxPoolD2Test {
    val target get() = MaxPoolD2(poolSize = 2, channel = 2, inputSize = 4, padding = 0)
    val input
        get() = Batch.of(
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
    fun `expect=指定区間内での最大値を取得`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(1f, 3f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(2f, 6f), actual = actual[0][1])

        assertContentEquals(expected = IOType.d1(0f, 2f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(0.3f, 0.9f), actual = actual[1][1])
    }

    @Test
    fun `train=勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 1f, 0f, 3f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 2f, 0f, 6f), actual = actual[0][1])

        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 2f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(0f, 0.3f, 0f, 0.9f), actual = actual[1][1])
    }
}
