@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.affine

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.assertContentEquals
import com.wsr.network.networkTestRule
import com.wsr.network.optimizer.Scheduler
import com.wsr.network.optimizer.sgd.Sgd
import com.wsr.network.process.Context
import kotlin.test.Test
import kotlin.test.assertEquals

class AffineD2Test {
    val target
        get() = AffineD2(
            channel = 2,
            outputSize = 4,
            optimizer = Sgd(scheduler = Scheduler.Fix(rate = 0.01f)).d2(3, 4),
            weight = IOType.d2(3, 4) { i, j -> i * 2f + j },
        )
    val input
        get() = batchOf(
            IOType.d2(
                IOType.d1(3) { it * 2f },
                IOType.d1(3) { it * 3f },
            ),
            IOType.d2(
                IOType.d1(3) { it * 4f },
                IOType.d1(3) { it * 5f },
            ),
        )

    @Test
    fun `expect=全結合`() = networkTestRule {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(20f, 26f, 32f, 38f), actual = actual[0][0])
        assertEquals(expected = IOType.d1(30f, 39f, 48f, 57f), actual = actual[0][1])
        assertEquals(expected = IOType.d1(40f, 52f, 64f, 76f), actual = actual[1][0])
        assertEquals(expected = IOType.d1(50f, 65f, 80f, 95f), actual = actual[1][1])
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() = networkTestRule {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(204f, 436f, 668f), actual = actual[0][0])
        assertEquals(expected = IOType.d1(306f, 654f, 1002f), actual = actual[0][1])
        assertEquals(expected = IOType.d1(408f, 872f, 1336f), actual = actual[1][0])
        assertEquals(expected = IOType.d1(510f, 1090f, 1670f), actual = actual[1][1])
    }

    @Test
    fun `train=重みを更新する`() = networkTestRule {
        val target = target

        target._train(input = input, context = Context(input), calcDelta = { it })
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(-7.0000f, -9.1f, -11.2000f, -13.3000f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-10.5000f, -13.6500f, -16.8000f, -19.95f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-14.0000f, -18.2f, -22.3999f, -26.6000f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-17.5000f, -22.7500f, -27.9999f, -33.2500f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
