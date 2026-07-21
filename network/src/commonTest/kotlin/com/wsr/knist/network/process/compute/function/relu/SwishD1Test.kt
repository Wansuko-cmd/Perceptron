@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.relu

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.GraphEnv
import kotlin.math.exp
import kotlin.test.Test

class SwishD1Test {
    val target
        get() = SwishD1(inputI = 4)
    val input
        get() = Batch.of(
            IOType.d1(0f, 1f, -1f, 2f),
            IOType.d1(0.5f, -0.5f, 1.5f, -2f),
        )

    fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))
    fun swish(x: Float): Float = x * sigmoid(x)

    @Test
    fun `expect=x_times_sigmoid_xを適用する`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d0(swish(0f)), actual = actual[0][0])
        assertContentEquals(expected = IOType.d0(swish(1f)), actual = actual[0][1])
        assertContentEquals(expected = IOType.d0(swish(-1f)), actual = actual[0][2])
        assertContentEquals(expected = IOType.d0(swish(2f)), actual = actual[0][3])
    }

    @Test
    fun `train=output_plus_sigmoid_times_1_minus_outputを勾配に掛けて返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D1>

        fun grad(x: Float): Float {
            val out = swish(x)
            val s = sigmoid(x)
            return (out + s * (1f - out)) * out
        }

        assertContentEquals(expected = IOType.d0(grad(0.5f)), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d0(grad(-0.5f)), actual = actual[1][1], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d0(grad(1.5f)), actual = actual[1][2], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d0(grad(-2f)), actual = actual[1][3], absoluteTolerance = 1e-4f)
    }
}
