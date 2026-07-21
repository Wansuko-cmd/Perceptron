@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.function.softmax

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.math.exp
import kotlin.test.Test

class SoftmaxD2Test {
    val target
        get() = SoftmaxD2(inputI = 2, inputJ = 2)

    val input
        get() = Batch.of(
            IOType.d2(2, 2) { i, j -> floatArrayOf(1f, 2f, -1f, 0.5f)[i * 2 + j] },
            IOType.d2(2, 2) { i, j -> floatArrayOf(0f, -0.5f, 1.5f, 2f)[i * 2 + j] },
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
    fun `expect=iとjの両方を含む全要素に対するsoftmaxを計算する`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        val e0 = softmax(floatArrayOf(1f, 2f, -1f, 0.5f))
        val e1 = softmax(floatArrayOf(0f, -0.5f, 1.5f, 2f))

        assertContentEquals(expected = IOType.d1(e0[0], e0[1]), actual = actual[0][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e0[2], e0[3]), actual = actual[0][1], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e1[0], e1[1]), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e1[2], e1[3]), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=softmaxのヤコビアンによるvjpを返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        val e0 = vjp(floatArrayOf(1f, 2f, -1f, 0.5f))
        val e1 = vjp(floatArrayOf(0f, -0.5f, 1.5f, 2f))

        assertContentEquals(expected = IOType.d1(e0[0], e0[1]), actual = actual[0][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e0[2], e0[3]), actual = actual[0][1], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e1[0], e1[1]), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(e1[2], e1[3]), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }
}
