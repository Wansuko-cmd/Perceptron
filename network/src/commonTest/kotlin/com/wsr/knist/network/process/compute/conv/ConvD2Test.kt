@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.conv

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.core.get
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import kotlin.test.Test

class ConvD2Test {
    val target
        get() = ConvD2(
            filter = 2,
            channel = 2,
            kernel = 2,
            stride = 1,
            dilation = 1,
            padding = 0,
            height = 3,
            width = 3,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d4(i = 2, j = 2, k = 2, l = 2),
            weight = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l },
        )
    val input
        get() = Batch.of(
            IOType.d3(2, 3, 3) { i, j, k -> i * 4f + j * 2f + k },
            IOType.d3(2, 3, 3) { i, j, k -> i * 6f + j * 3f + k },
        )

    @Test
    fun `expect=2次元畳み込み`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(138f, 194f),
                IOType.d1(166f, 222f),
            ),
            actual = actual[0][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(362f, 546f),
                IOType.d1(454f, 638f),
            ),
            actual = actual[0][1],
        )

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(198f, 282f),
                IOType.d1(226f, 310f),
            ),
            actual = actual[1][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(518f, 794f),
                IOType.d1(610f, 886f),
            ),
            actual = actual[1][1],
        )
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(2896f, 7528f, 4872f),
                IOType.d1(7764f, 19600f, 12316f),
                IOType.d1(5108f, 12552f, 7684f),
            ),
            actual = actual[0][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(4896f, 12008f, 7352f),
                IOType.d1(12724f, 30480f, 18236f),
                IOType.d1(8068f, 18952f, 11124f),
            ),
            actual = actual[0][1],
        )

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(4144f, 10456f, 6552f),
                IOType.d1(11212f, 27600f, 16868f),
                IOType.d1(7428f, 17864f, 10676f),
            ),
            actual = actual[1][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(7008f, 16664f, 9896f),
                IOType.d1(18380f, 42896f, 24996f),
                IOType.d1(11732f, 26952f, 15460f),
            ),
            actual = actual[1][1],
        )
    }

    @Test
    fun `train=重みを更新する`() = networkScopeTestRule {
        val target = target

        with(target) { _train(input = input, env = GraphEnv(), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(-1886.6399f, -2721.2f),
                IOType.d1(-2303.92f, -3138.48f),
            ),
            actual = actual[0][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(-5273.52f, -7572.4f),
                IOType.d1(-6422.96f, -8721.84f),
            ),
            actual = actual[0][1],
        )

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(-2718.96f, -3970.7998f),
                IOType.d1(-3136.24f, -4388.08f),
            ),
            actual = actual[1][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(-7600.88f, -11049.2f),
                IOType.d1(-8750.32f, -12198.641f),
            ),
            actual = actual[1][1],
        )
    }
}
