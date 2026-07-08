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
import com.wsr.knist.network.process.compute.norm.layer.d3.LayerNormAxisD3
import kotlin.test.Test

class LayerNormAxisD3Test {
    val target0 get() = LayerNormAxisD3(inputI = 2, inputJ = 2, inputK = 3, axis = 0, e = 1e-6f)
    val target1 get() = LayerNormAxisD3(inputI = 2, inputJ = 2, inputK = 3, axis = 1, e = 1e-6f)
    val target2 get() = LayerNormAxisD3(inputI = 2, inputJ = 2, inputK = 3, axis = 2, e = 1e-6f)
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
    fun `Axis0_expect=axis0で層正規化`() = networkScopeTestRule {
        val actual = with(target0) { _expect(input = input, context = Context(input)) } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.9999f, 0.9999f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.9999f, 0.9999f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, -0.9999f, -0.9999f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, -0.9999f, -0.9999f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.9999f, 0.9999f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.9999f, 1.0000f, 1.0000f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, -0.9999f, -0.9999f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.9999f, -1.0000f, -1.0000f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis0_train=axis0で正規化および勾配を伝播`() = networkScopeTestRule {
        val actual = with(target0) {
            _train(
                input = input,
                context = Context(input),
                calcDelta = { 1e6f * it as Batch<IOType.D2> },
            )
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(0.0000f, 8.1250f, 8.1250f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 1.5000f, 0.2500f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, -8.1250f, -8.1250f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, -1.6250f, -0.3125f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0.0000f, 8.1250f, 0.0312f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0312f, 0.0156f, 0.0000f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, -8.1250f, -0.0312f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.0312f, -0.0156f, 0.0000f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis1_expect=axis1で層正規化`() = networkScopeTestRule {
        val actual = with(target1) { _expect(input = input, context = Context(input)) } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(0.0000f, -0.9999f, -0.9999f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.9999f, 0.9999f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, -0.9999f, 0.9999f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.9999f, -0.9999f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-0.9999f, -0.9999f, 0.9999f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.9999f, 0.9999f, -0.9999f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.9999f, 0.9999f, 0.9999f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.9999f, -1.0000f, -1.0000f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis1_train=axis1で正規化および勾配を伝播`() = networkScopeTestRule {
        val actual = with(target1) {
            _train(
                input = input,
                context = Context(input),
                calcDelta = { it },
            )
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(0.0000f, -8.1062e-6f, -9.5367e-7f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-6f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 8.1062e-6f, 9.5367e-7f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-6f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, -2.9659e-4f, 1.2540e-4f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 2.9659e-4f, -1.2540E-4f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-1.1920e-7f, -1.1920e-7f, 8.1062e-6f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-6f,
        )
        assertContentEquals(
            expected = IOType.d1(1.1920e-7f, 1.1920e-7f, -8.1062e-6f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-7f,
        )
        assertContentEquals(
            expected = IOType.d1(9.5367e-7f, 1.7881e-7f, 1.1920e-7f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-7f,
        )
        assertContentEquals(
            expected = IOType.d1(-9.5367e-7f, -1.7881e-7f, -1.1920e-7f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-7f,
        )
    }

    @Test
    fun `Axis2_expect=axis2で層正規化`() = networkScopeTestRule {
        val actual = with(target2) { _expect(input = input, context = Context(input)) } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0.0000f, 1.2247f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0.0000f, 1.2247f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.7071f, -0.7071f, 1.4142f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0.0000f, 1.2247f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-0.9805f, -0.3922f, 1.3728f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0.0000f, 1.2247f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.2247f, 1.2247f, 0.0000f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(1.1111f, 0.2020f, -1.3131f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis2_train=axis2で正規化および勾配を伝播`() = networkScopeTestRule {
        val actual = with(target2) {
            _train(
                input = input,
                context = Context(input),
                calcDelta = { 1e6f * it as Batch<IOType.D2> },
            )
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(-2.2500f, 0.0000f, 2.2500f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.1875f, 0.0000f, 0.1875f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-6.6250f, -6.6250f, 13.2500f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-82.5000f, 0.0000f, 82.5000f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-0.0312f, -0.0156f, 0.0000f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-2.2500f, 0.0000f, 2.2500f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-17.7500f, 17.7500f, 0.0000f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(6.8750f, 1.3125f, -8.2500f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
