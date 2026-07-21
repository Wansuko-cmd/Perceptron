@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.softmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.GraphEnv
import kotlin.math.exp
import kotlin.test.Test

class SoftmaxD3Test {
    val target
        get() = SoftmaxD3(inputI = 1, inputJ = 2, inputK = 2)

    val input
        get() = Batch.of(
            IOType.d3(1, 2, 2) { _, j, k -> floatArrayOf(1f, 2f, -1f, 0.5f)[j * 2 + k] },
            IOType.d3(1, 2, 2) { _, j, k -> floatArrayOf(0f, -0.5f, 1.5f, 2f)[j * 2 + k] },
        )

    private fun softmax(xs: FloatArray): FloatArray {
        val max = xs.max()
        val exp = FloatArray(xs.size) { exp(xs[it] - max) }
        val sum = exp.sum()
        return FloatArray(xs.size) { exp[it] / sum }
    }

    private fun vjp(xs: FloatArray): FloatArray {
        val out = softmax(xs)
        val sum = out.sumOf { (it * it).toDouble() }.toFloat()
        return FloatArray(out.size) { out[it] * (out[it] - sum) }
    }

    @Test
    fun `expect=i_j_kすべてを含む全要素に対するsoftmaxを計算する`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        val e0 = softmax(floatArrayOf(1f, 2f, -1f, 0.5f))
        val e1 = softmax(floatArrayOf(0f, -0.5f, 1.5f, 2f))

        assertContentEquals(expected = IOType.d1(e0[0], e0[1]), actual = actual[0][0][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e0[2], e0[3]), actual = actual[0][0][1], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e1[0], e1[1]), actual = actual[1][0][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e1[2], e1[3]), actual = actual[1][0][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=softmaxのヤコビアンによるvjpを返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D3>

        val e0 = vjp(floatArrayOf(1f, 2f, -1f, 0.5f))
        val e1 = vjp(floatArrayOf(0f, -0.5f, 1.5f, 2f))

        assertContentEquals(expected = IOType.d1(e0[0], e0[1]), actual = actual[0][0][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e0[2], e0[3]), actual = actual[0][0][1], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e1[0], e1[1]), actual = actual[1][0][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e1[2], e1[3]), actual = actual[1][0][1], absoluteTolerance = 1e-4f)
    }
}
