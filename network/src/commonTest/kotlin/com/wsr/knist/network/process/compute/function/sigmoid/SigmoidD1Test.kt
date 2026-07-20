@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.sigmoid

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.Context
import kotlin.math.exp
import kotlin.test.Test

class SigmoidD1Test {
    val target
        get() = SigmoidD1(inputI = 4)
    val input
        get() = Batch.of(
            IOType.d1(0f, 1f, -1f, 2f),
            IOType.d1(0.5f, -0.5f, 1.5f, -2f),
        )

    fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))

    @Test
    fun `expect=シグモイド関数を適用する`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d0(sigmoid(0f)), actual = actual[0][0])
        assertContentEquals(expected = IOType.d0(sigmoid(1f)), actual = actual[0][1])
        assertContentEquals(expected = IOType.d0(sigmoid(-1f)), actual = actual[0][2])
        assertContentEquals(expected = IOType.d0(sigmoid(2f)), actual = actual[0][3])
    }

    @Test
    fun `train=output_times_1_minus_outputを勾配に掛けて返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D1>

        fun grad(x: Float): Float {
            val s = sigmoid(x)
            return s * s * (1f - s)
        }

        assertContentEquals(expected = IOType.d0(grad(0.5f)), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d0(grad(-0.5f)), actual = actual[1][1], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d0(grad(1.5f)), actual = actual[1][2], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d0(grad(-2f)), actual = actual[1][3], absoluteTolerance = 1e-4f)
    }
}
