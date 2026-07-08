@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.norm.layer

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.norm.layer.d3.LayerNormD3
import kotlin.test.Test

class LayerNormD3Test {
    val target get() = LayerNormD3(inputI = 2, inputJ = 2, inputK = 3, e = 1e-6f)
    val input
        get() = Batch.of(
            IOType.d3(
                IOType.d2(
                    IOType.d1(3) { it.toFloat() },
                    IOType.d1(3) { it * 2f },
                ),
                IOType.d2(
                    IOType.d1(3) { 10f % (it + 1) },
                    IOType.d1(3) { it * 0.3f },
                ),
            ),
            IOType.d3(
                IOType.d2(
                    IOType.d1(3) { it * it * 2f },
                    IOType.d1(3) { it + 5f },
                ),
                IOType.d2(
                    IOType.d1(3) { it % 1.5f },
                    IOType.d1(3) { 10f / (it - 5) },
                ),
            ),
        )

    @Test
    fun `expect=層正規化`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(-0.7734f, 0.0780f, 0.9295f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.7734f, 0.9295f, 2.6326f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.7734f, -0.7734f, 0.0780f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.7734f, -0.5180f, -0.2625f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-0.4921f, 0.0530f, 1.6885f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.8707f, 1.1433f, 1.4159f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.4921f, -0.2195f, -0.3558f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.0373f, -1.1736f, -1.4008f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=正規化および勾配を伝播`() = networkScopeTestRule {
        val actual = with(target) {
            _train(
                input = input,
                context = Context(input),
                calcDelta = { 1e6f * it as Batch<IOType.D2> },
            )
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(658641.7500f, -66468.4300f, -791578.6000f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(658641.7500f, -791578.6000f, -2241799.0000f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(658641.7500f, 658641.7500f, -66468.4300f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(658641.7500f, 441108.7000f, 223575.6400f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(134166.4500f, -14448.6860f, -460294.1600f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-237371.4200f, -311679.0000f, -385986.5600f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(134166.4500f, 59858.8900f, 97012.6800f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(282781.6000f, 319935.3800f, 381858.3800f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
