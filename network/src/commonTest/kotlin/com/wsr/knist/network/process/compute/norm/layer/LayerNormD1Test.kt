@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.norm.layer

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.compute.norm.layer.d1.LayerNormD1
import kotlin.test.Test

class LayerNormD1Test {
    val target get() = LayerNormD1(inputI = 3, e = 1e-6f)
    val input
        get() = Batch.of(
            IOType.d1(3) { it.toFloat() },
            IOType.d1(3) { it * 2f },
        )

    @Test
    fun `expect=層正規化`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D1>

        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0f, 1.2247f),
            actual = actual[0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0f, 1.2247f),
            actual = actual[1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=正規化および勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(
                input = input,
                env = GraphEnv(),
                calcDelta = { 1e6f * it as Batch<IOType.D1> },
            )
        } as Batch<IOType.D1>

        assertContentEquals(
            expected = IOType.d1(-2.2500f, 0.0000f, 2.2500f),
            actual = actual[0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.1875f, 0.0000f, 0.1875f),
            actual = actual[1],
            absoluteTolerance = 1e-4f,
        )
    }
}
