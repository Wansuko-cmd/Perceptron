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
import com.wsr.knist.network.process.GraphEnv
import kotlin.test.Test

class MaxPoolD2Test {
    val target get() = MaxPoolD2(poolSize = 2, channel = 2, inputSize = 4, padding = 0, stride = 2)
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
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(1f, 3f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(2f, 6f), actual = actual[0][1])

        assertContentEquals(expected = IOType.d1(0f, 2f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(0.3f, 0.9f), actual = actual[1][1])
    }

    @Test
    fun `train=勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 1f, 0f, 3f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 2f, 0f, 6f), actual = actual[0][1])

        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 2f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(0f, 0.3f, 0f, 0.9f), actual = actual[1][1])
    }

    val strideTarget get() = MaxPoolD2(poolSize = 2, channel = 2, inputSize = 4, padding = 0, stride = 1)
    val strideInput
        get() = Batch.of(
            IOType.d2(
                IOType.d1(1f, 3f, 2f, 0f),
                IOType.d1(4f, 1f, 5f, 2f),
            ),
            IOType.d2(
                IOType.d1(0f, 2f, 3f, 1f),
                IOType.d1(5f, 0f, 1f, 4f),
            ),
        )

    @Test
    fun `expect=strideが窓幅未満の時は重複した窓で最大値を取得`() = networkScopeTestRule {
        val actual = with(strideTarget) {
            _expect(input = strideInput, env = GraphEnv())
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(3f, 3f, 2f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(4f, 5f, 5f), actual = actual[0][1])

        assertContentEquals(expected = IOType.d1(2f, 3f, 3f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(5f, 1f, 4f), actual = actual[1][1])
    }

    @Test
    fun `train=strideが窓幅未満の時は重複した窓の勾配を加算して伝播`() = networkScopeTestRule {
        val actual = with(strideTarget) {
            _train(input = strideInput, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 6f, 2f, 0f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(4f, 0f, 10f, 0f), actual = actual[0][1])

        assertContentEquals(expected = IOType.d1(0f, 2f, 6f, 0f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(5f, 0f, 1f, 4f), actual = actual[1][1])
    }
}
