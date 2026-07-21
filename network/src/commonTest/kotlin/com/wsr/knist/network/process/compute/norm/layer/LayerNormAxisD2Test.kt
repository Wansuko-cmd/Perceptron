@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.norm.layer

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.process.GraphEnv
import com.wsr.knist.network.process.compute.norm.layer.d2.LayerNormAxisD2
import kotlin.test.Test

class LayerNormAxisD2Test {
    val target0 get() = LayerNormAxisD2(inputI = 2, inputJ = 3, axis = 0, e = 1e-6f)
    val target1 get() = LayerNormAxisD2(inputI = 2, inputJ = 3, axis = 1, e = 1e-6f)
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
    fun `Axis0_expect=axis0で層正規化`() = networkScopeTestRule {
        val actual = with(target0) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(0.0000f, -0.9999f, -0.9999f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.9999f, 0.9999f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0.0000f, -0.9999f, 0.9999f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 0.9999f, -0.9999f),
            actual = actual[1][1],
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
        } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(0.0000f, -8.1250f, -0.9375f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 8.1250f, 0.9375f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(0.0000f, -296.5000f, 125.0000f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(0.0000f, 296.5000f, -125.0000f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis1_expect=axis1で層正規化`() = networkScopeTestRule {
        val actual = with(target1) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0.0000f, 1.2247f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0.0000f, 1.2247f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-0.7071f, -0.7071f, 1.4142f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-1.2247f, 0.0000f, 1.2247f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `Axis1_train=axis1で正規化および勾配を伝播`() = networkScopeTestRule {
        val actual = with(target1) {
            _train(
                input = input,
                env = GraphEnv(),
                calcDelta = { 1e6f * it as Batch<IOType.D2> },
            )
        } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(-2.2500f, 0.0000f, 2.2500f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-0.1875f, 0.0000f, 0.1875f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-6.6250f, -6.6250f, 13.2500f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-82.5000f, 0.0000f, 82.5000f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
