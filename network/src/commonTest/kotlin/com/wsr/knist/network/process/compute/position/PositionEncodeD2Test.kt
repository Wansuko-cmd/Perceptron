@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.position

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class PositionEncodeD2Test {
    val target get() = PositionEncodeD2(inputI = 2, inputJ = 3, waveLength = 100f)
    val input
        get() = Batch.of(
            IOType.d2(
                IOType.d1(3) { it.toFloat() },
                IOType.d1(3) { it * 2f },
            ),
            IOType.d2(
                IOType.d1(3) { 10f % (it + 1) },
                IOType.d1(3) { it * 0.3f },
            ),
        )

    @Test
    fun `expect=位置情報埋め込み`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 2f, 2f), actual = actual[0][0])
        assertContentEquals(
            expected = IOType.d1(0.8414f, 2.5403f, 4.0463f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(expected = IOType.d1(0f, 1f, 1f), actual = actual[1][0])
        assertContentEquals(
            expected = IOType.d1(0.8414f, 0.8403f, 0.6463f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, env = GraphEnv(), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 2f, 2f), actual = actual[0][0])
        assertContentEquals(
            expected = IOType.d1(0.8414f, 2.5403f, 4.0463f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(expected = IOType.d1(0f, 1f, 1f), actual = actual[1][0])
        assertContentEquals(
            expected = IOType.d1(0.8414f, 0.8403f, 0.6463f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
