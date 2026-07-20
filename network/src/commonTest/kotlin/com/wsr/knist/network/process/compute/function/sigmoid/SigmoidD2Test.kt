@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.sigmoid

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.Context
import kotlin.math.exp
import kotlin.test.Test

class SigmoidD2Test {
    val target
        get() = SigmoidD2(inputI = 2, inputJ = 2)
    val input
        get() = Batch.of(
            IOType.d2(2, 2) { i, j -> if (i == 0) j.toFloat() else -j.toFloat() - 1f },
            IOType.d2(2, 2) { i, j -> if (i == 0) j * 0.5f else -j * 1.5f - 2f },
        )

    fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))

    @Test
    fun `expect=シグモイド関数を適用する`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(sigmoid(0f), sigmoid(1f)),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(sigmoid(-1f), sigmoid(-2f)),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(sigmoid(0f), sigmoid(0.5f)),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(sigmoid(-2f), sigmoid(-3.5f)),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=output_times_1_minus_outputを勾配に掛けて返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D2>

        fun grad(x: Float): Float {
            val s = sigmoid(x)
            return s * s * (1f - s)
        }

        assertContentEquals(expected = IOType.d1(grad(0f), grad(1f)), actual = actual[0][0], absoluteTolerance = 1e-4f)
        assertContentEquals(
            expected = IOType.d1(grad(-1f), grad(-2f)),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(grad(0f), grad(0.5f)),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(grad(-2f), grad(-3.5f)),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
