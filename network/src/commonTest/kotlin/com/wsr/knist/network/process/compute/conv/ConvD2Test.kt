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
                IOType.d1(140f, 168f),
                IOType.d1(196f, 224f),
            ),
            actual = actual[0][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(364f, 456f),
                IOType.d1(548f, 640f),
            ),
            actual = actual[0][1],
        )

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(202f, 230f),
                IOType.d1(286f, 314f),
            ),
            actual = actual[1][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(522f, 614f),
                IOType.d1(798f, 890f),
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
                IOType.d1(2912f, 7064f, 4272f),
                IOType.d1(8304f, 19568f, 11504f),
                IOType.d1(5872f, 13464f, 7712f),
            ),
            actual = actual[0][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(4928f, 11576f, 6768f),
                IOType.d1(13296f, 30512f, 17456f),
                IOType.d1(8848f, 19896f, 11168f),
            ),
            actual = actual[0][1],
        )

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(4176f, 9812f, 5756f),
                IOType.d1(12008f, 27536f, 15768f),
                IOType.d1(8552f, 19164f, 10732f),
            ),
            actual = actual[1][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(7072f, 16084f, 9132f),
                IOType.d1(19240f, 42960f, 23960f),
                IOType.d1(12888f, 28316f, 15548f),
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
                IOType.d1(-1912.5599f, -2335.92f),
                IOType.d1(-2759.2798f, -3182.6401f),
            ),
            actual = actual[0][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(-5299.44f, -6454.9595f),
                IOType.d1(-7610.4795f, -8766.0f),
            ),
            actual = actual[0][1],
        )

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(-2755.1997f, -3178.56f),
                IOType.d1(-4025.2798f, -4448.6396f),
            ),
            actual = actual[1][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(-7637.1196f, -8792.641f),
                IOType.d1(-11103.68f, -12259.199f),
            ),
            actual = actual[1][1],
        )
    }
}
