@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.norm.layer

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.process.compute.norm.layer.d2.LayerNormD2
import kotlin.test.Test

class LayerNormD2Test {
    val target get() = LayerNormD2(inputI = 2, inputJ = 3, e = 1e-6f)
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
    fun `expect=層正規化`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(-1.0834f, -0.3611f, 0.3611f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.0834f, 0.3611f, 1.8057f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-0.8421f, -0.8421f, 1.8172f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.8421f, -0.0443f, 0.7535f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=正規化および勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(
                input = input,
                env = GraphEnv(),
                calcDelta = { 1e6f * it as Batch<IOType.D2> },
            )
        } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(-0.3125f, -0.1406f, 0.1093f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.3125f, 0.1093f, 0.6250f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-16.0000f, -16.0000f, 34.5000f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-16.0000f, -0.9453f, 14.1250f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
