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
import com.wsr.knist.network.GraphEnv
import kotlin.test.Test

class MaxPoolD3Test {
    val target get() = MaxPoolD3(poolSize = 2, channel = 2, height = 2, width = 4, padding = 0, stride = 2)
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
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(2f, 6f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(0.3f, 2f), actual = actual[0][1][0])

        assertContentEquals(expected = IOType.d1(6f, 18f), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(1f, 0.5f), actual = actual[1][1][0])
    }

    @Test
    fun `train=勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 0f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 2f), actual = actual[0][1][0])

        assertContentEquals(expected = IOType.d1(0f, 0f, 0f, 18f), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(0f, 1f, 0.5f, 0f), actual = actual[1][1][0])
    }

    val strideTarget get() = MaxPoolD3(poolSize = 2, channel = 2, height = 3, width = 3, padding = 0, stride = 1)
    val strideInput
        get() = Batch.of(
            IOType.d3(
                IOType.d2(
                    IOType.d1(1f, 2f, 0f),
                    IOType.d1(3f, 5f, 1f),
                    IOType.d1(0f, 2f, 4f),
                ),
                IOType.d2(
                    IOType.d1(4f, 0f, 1f),
                    IOType.d1(0f, 2f, 0f),
                    IOType.d1(3f, 0f, 5f),
                ),
            ),
            IOType.d3(
                IOType.d2(
                    IOType.d1(9f, 1f, 2f),
                    IOType.d1(1f, 3f, 1f),
                    IOType.d1(2f, 1f, 8f),
                ),
                IOType.d2(
                    IOType.d1(1f, 2f, 3f),
                    IOType.d1(4f, 5f, 6f),
                    IOType.d1(7f, 8f, 9f),
                ),
            ),
        )

    @Test
    fun `expect=strideが窓幅未満の時は重複した窓で最大値を取得`() = networkScopeTestRule {
        val actual = with(strideTarget) {
            _expect(input = strideInput, env = GraphEnv())
        } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(5f, 5f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(5f, 5f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(4f, 2f), actual = actual[0][1][0])
        assertContentEquals(expected = IOType.d1(3f, 5f), actual = actual[0][1][1])

        assertContentEquals(expected = IOType.d1(9f, 3f), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(3f, 8f), actual = actual[1][0][1])
        assertContentEquals(expected = IOType.d1(5f, 6f), actual = actual[1][1][0])
        assertContentEquals(expected = IOType.d1(8f, 9f), actual = actual[1][1][1])
    }

    @Test
    fun `train=strideが窓幅未満の時は重複した窓の勾配を加算して伝播`() = networkScopeTestRule {
        val actual = with(strideTarget) {
            _train(input = strideInput, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D3>

        assertContentEquals(expected = IOType.d1(0f, 0f, 0f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(0f, 20f, 0f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(0f, 0f, 0f), actual = actual[0][0][2])
        assertContentEquals(expected = IOType.d1(4f, 0f, 0f), actual = actual[0][1][0])
        assertContentEquals(expected = IOType.d1(0f, 2f, 0f), actual = actual[0][1][1])
        assertContentEquals(expected = IOType.d1(3f, 0f, 5f), actual = actual[0][1][2])

        assertContentEquals(expected = IOType.d1(9f, 0f, 0f), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(0f, 6f, 0f), actual = actual[1][0][1])
        assertContentEquals(expected = IOType.d1(0f, 0f, 8f), actual = actual[1][0][2])
        assertContentEquals(expected = IOType.d1(0f, 0f, 0f), actual = actual[1][1][0])
        assertContentEquals(expected = IOType.d1(0f, 5f, 6f), actual = actual[1][1][1])
        assertContentEquals(expected = IOType.d1(0f, 8f, 9f), actual = actual[1][1][2])
    }
}
