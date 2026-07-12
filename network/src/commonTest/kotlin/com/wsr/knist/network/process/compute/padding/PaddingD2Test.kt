@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.padding

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.Context
import kotlin.test.Test

class PaddingD2Test {
    val input
        get() = Batch.of(
            IOType.d2(2, 3) { i, j -> i * 3f + j },
            IOType.d2(2, 3) { i, j -> i * 5f + j + 1f },
        )

    @Test
    fun `expect=axis=1_leftとrightにゼロが挿入される`() = networkScopeTestRule {
        val target = PaddingD2(axis = 1, left = 2, right = 1, inputI = 2, inputJ = 3)

        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 1f, 2f, 0f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 0f, 3f, 4f, 5f, 0f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, 0f, 1f, 2f, 3f, 0f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(0f, 0f, 6f, 7f, 8f, 0f), actual = actual[1][1])
    }

    @Test
    fun `expect=axis=0_channel方向にゼロが挿入される`() = networkScopeTestRule {
        val target = PaddingD2(axis = 0, left = 1, right = 0, inputI = 2, inputJ = 3)

        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 0f, 0f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 1f, 2f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(3f, 4f, 5f), actual = actual[0][2])

        assertContentEquals(expected = IOType.d1(0f, 0f, 0f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(1f, 2f, 3f), actual = actual[1][1])
        assertContentEquals(expected = IOType.d1(6f, 7f, 8f), actual = actual[1][2])
    }

    @Test
    fun `train=padding分の勾配を捨てて元の形状に戻す`() = networkScopeTestRule {
        val target = PaddingD2(axis = 1, left = 2, right = 1, inputI = 2, inputJ = 3)

        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 1f, 2f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(3f, 4f, 5f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(1f, 2f, 3f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(6f, 7f, 8f), actual = actual[1][1])
    }

    @Test
    fun `train=right=0のときも正しく元の形状に戻す`() = networkScopeTestRule {
        val target = PaddingD2(axis = 1, left = 2, right = 0, inputI = 2, inputJ = 3)

        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 1f, 2f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(3f, 4f, 5f), actual = actual[0][1])
    }
}
