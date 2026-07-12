@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.conv

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.process.Context
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
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D3>

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
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D3>

        println(actual[1])

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(4396f, 8888f, 4252f),
                IOType.d1(10484f, 20560f, 9596f),
                IOType.d1(5848f, 11192f, 5104f),
            ),
            actual = actual[0][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(6396f, 13368f, 6732f),
                IOType.d1(15444f, 31440f, 15516f),
                IOType.d1(8808f, 17592f, 8544f),
            ),
            actual = actual[0][1],
        )

        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(6292f, 12248f, 5716f),
                IOType.d1(15156f, 28800f, 13164f),
                IOType.d1(8504f, 15832f, 7088f),
            ),
            actual = actual[1][0],
        )
        assertContentEquals(
            expected = IOType.d2(
                IOType.d1(9156f, 18456f, 9060f),
                IOType.d1(22324f, 44096f, 21292f),
                IOType.d1(12808f, 24920f, 11872f),
            ),
            actual = actual[1][1],
        )
    }

    @Test
    fun `train=重みを更新する`() = networkScopeTestRule {
        val target = target

        with(target) { _train(input = input, context = Context(input), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D3>

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
