@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.scale.d2

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
import kotlin.test.assertContentEquals

class ScaleD2Test {
    val target
        get() = ScaleD2(
            outputX = 2,
            outputY = 2,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d2(2, 2),
            weight = IOType.d2(2, 2) { i, j -> i * 2f + j },
        )

    val input
        get() = batchOf(
            IOType.d2(
                IOType.d1(2) { it * 2f },
                IOType.d1(2) { it * 3f },
            ),
            IOType.d2(
                IOType.d1(2) { it * -2f },
                IOType.d1(2) { it * -1f },
            ),
        )

    @Test
    fun `expect=スケール項`() = networkTestRule {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 9f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, -2f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -3f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() = networkTestRule {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 2f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 27f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(0f, -2f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -9f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `train=重みを更新する`() = networkTestRule {
        val target = target

        target._train(input = input, context = Context(input), calcDelta = { it })
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 1.92f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(0f, 8.5499f), actual = actual[0][1], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -1.92f), actual = actual[1][0], absoluteTolerance = 1e-4f)
        assertContentEquals(expected = IOType.d1(0f, -2.85f), actual = actual[1][1], absoluteTolerance = 1e-4f)
    }
}
