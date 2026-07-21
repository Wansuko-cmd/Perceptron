@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.norm.rms

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.process.compute.norm.rms.d3.RmsNormAxisD3
import kotlin.test.Test

class RmsNormAxisD3Test {
    val target0 get() = RmsNormAxisD3(inputI = 2, inputJ = 2, inputK = 3, axis = 0, e = 1e-6f)
    val target1 get() = RmsNormAxisD3(inputI = 2, inputJ = 2, inputK = 3, axis = 1, e = 1e-6f)
    val target2 get() = RmsNormAxisD3(inputI = 2, inputJ = 2, inputK = 3, axis = 2, e = 1e-6f)
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
        val actual = with(target0) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(0.0000f, 1.4142f, 1.2649f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 1.3985f, 1.3985f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 0.6324f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.2097f, 0.2097f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0.0000f, 1.2649f, 1.4114f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(1.3130f, 1.3054f, 1.2768f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.6324f, 0.0882f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.52529f, -0.5439f, -0.6080f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis0_train=axis0で正規化および勾配を伝播`() = networkScopeTestRule {
        val actual = with(target0) {
            _train(
                input = input,
                env = GraphEnv(),
                calcDelta = { 1e6f * it as Batch<IOType.D2> },
            )
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(0.0000f, 4.1250f, 0.2500f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.5625f, 0.1250f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 0.1250f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.0781f, 0.0234f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.2500f, -0.0156f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.0312f, 0.0312f, 0.0312f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.1250f, -0.0009f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0312f, -0.0078f, -0.0156f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis1_expect=axis1で層正規化`() = networkScopeTestRule {
        val actual = with(target1) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.6324f, 0.6324f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 1.2649f, 1.2649f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 1.2126f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 1.4141f, 0.7276f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.4472f, 1.0643f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(1.4142f, 1.3416f, 0.9312f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.5252f, 0.2097f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.4142f, -1.3130f, -1.3985f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis1_train=axis1で正規化および勾配を伝播`() = networkScopeTestRule {
        val actual = with(target1) {
            _train(
                input = input,
                env = GraphEnv(),
                calcDelta = { it },
            )
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(0.0000f, 1.1920E-7f, 1.4901E-8f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-6f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 2.3841E-7f, 2.9802E-8f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-6f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 2.1457E-6f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 1.4734E-4f, 1.2516E-6f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 0.0000f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-6f,
        )
        assertContentEquals(
            expected = IOType.d1(8.9406E-8f, 0.0000f, 7.4505E-9f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-7f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 8.9406E-8f, 2.2351E-8f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-7f,
        )
        assertContentEquals(
            expected = IOType.d1(-5.3644E-7f, -2.3841E-7f, -1.1920E-7f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-7f,
        )
    }

    @Test
    fun `Axis2_expect=axis2で層正規化`() = networkScopeTestRule {
        val actual = with(target2) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.7745f, 1.5491f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.7745f, 1.5491f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 1.7320f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.7745f, 1.5491f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.4200f, 1.6803f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.8257f, 0.9908f, 1.1560f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 1.5491f, 0.7745f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.7495f, -0.9368f, -1.2491f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis2_train=axis2で正規化および勾配を伝播`() = networkScopeTestRule {
        val actual = with(target2) {
            _train(
                input = input,
                env = GraphEnv(),
                calcDelta = { 1e6f * it as Batch<IOType.D2> },
            )
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.3125f, 0.6250f),
            actual = actual[0][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.0312f, 0.0625f),
            actual = actual[0][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.0000f, 9.0000f),
            actual = actual[0][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 13.1250f, 26.2500f),
            actual = actual[0][1][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.0078f, 0.0312f),
            actual = actual[1][0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0156f, 0.0000f, 0.0000f),
            actual = actual[1][0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 5.5000f, 2.7500f),
            actual = actual[1][1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.0625f, -0.0937f, -0.1250f),
            actual = actual[1][1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
