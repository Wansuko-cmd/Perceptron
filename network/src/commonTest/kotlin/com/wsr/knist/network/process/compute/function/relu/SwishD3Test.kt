@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.Context
import kotlin.math.exp
import kotlin.test.Test

class SwishD3Test {
    val target
        get() = SwishD3(inputI = 1, inputJ = 2, inputK = 2)
    val input
        get() = Batch.of(
            IOType.d3(1, 2, 2) { _, j, k -> if (j == 0) k.toFloat() else -k.toFloat() - 1f },
            IOType.d3(1, 2, 2) { _, j, k -> if (j == 0) k * 0.5f else -k * 1.5f - 2f },
        )

    fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))
    fun swish(x: Float): Float = x * sigmoid(x)

    @Test
    fun `expect=x_times_sigmoid_xを適用する`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(swish(0f), swish(1f)),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(swish(-1f), swish(-2f)),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(swish(0f), swish(0.5f)),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(swish(-2f), swish(-3.5f)),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=output_plus_sigmoid_times_1_minus_outputを勾配に掛けて返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D3>

        fun grad(x: Float): Float {
            val out = swish(x)
            val s = sigmoid(x)
            return (out + s * (1f - out)) * out
        }

        assertContentEquals(
            expected = IOType.d1(grad(0f), grad(1f)),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(grad(-1f), grad(-2f)),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(grad(0f), grad(0.5f)),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(grad(-2f), grad(-3.5f)),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
