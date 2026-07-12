@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.conv

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.process.Context
import kotlin.test.Test

class ConvD1Test {
    val target
        get() = ConvD1(
            filter = 2,
            channel = 3,
            kernel = 2,
            stride = 1,
            dilation = 1,
            padding = 0,
            inputSize = 4,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d3(i = 2, j = 3, k = 2),
            weight = IOType.d3(2, 3, 2) { i, j, k -> i * 4f + j * 2f + k },
        )
    val input
        get() = Batch.of(
            IOType.d2(3, 4) { i, j -> i * 2f + j },
            IOType.d2(3, 4) { i, j -> i * 3f + j },
        )

    @Test
    fun `expect=1次元畳み込み`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(55f, 70f, 85f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(115f, 154f, 193f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(78f, 93f, 108f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(162f, 201f, 240f), actual = actual[1][1])
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(630f, 1300f, 1666f, 772f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(970f, 2088f, 2670f, 1328f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(888f, 1746f, 2112f, 960f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(1368f, 2814f, 3396f, 1656f), actual = actual[1][1])
    }

    @Test
    fun `train=重みを更新する`() = networkScopeTestRule {
        val target = target

        with(target) { _train(input = input, context = Context(input), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(-152.735f, -199.25f, -245.76498f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-338.7949f, -434.3899f, -529.985f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-216.39f, -262.9049f, -309.4199f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-481.05f, -576.6449f, -672.24f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
