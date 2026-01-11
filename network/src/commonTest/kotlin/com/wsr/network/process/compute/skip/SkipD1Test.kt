@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.skip

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.network.NetworkTestRule
import com.wsr.optimizer.Scheduler
import com.wsr.optimizer.sgd.Sgd
import com.wsr.process.Context
import com.wsr.process.compute.bias.d1.BiasD1
import com.wsr.process.compute.skip.SkipD1
import org.junit.Rule
import kotlin.test.Test
import kotlin.test.assertEquals

class SkipD1Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target = SkipD1(
        layers = listOf(
            BiasD1(
                outputSize = 3,
                optimizer = Sgd(Scheduler.Fix(rate = 0.01f)).d1(3),
                weight = IOType.d1(3) { it.toFloat() },
            ),
            BiasD1(
                outputSize = 3,
                optimizer = Sgd(Scheduler.Fix(rate = 0.01f)).d1(3),
                weight = IOType.d1(3) { it.toFloat() },
            ),
        ),
        inputSize = 3,
        outputSize = 3,
    )
    val input
        get() = batchOf(
            IOType.d1(3) { it * 2f },
            IOType.d1(3) { it * 3f },
        )

    @Test
    fun `expect=スキップ接続を行う`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D1>

        assertEquals(expected = IOType.d1(0f, 6f, 12f), actual = actual[0])
        assertEquals(expected = IOType.d1(0f, 8f, 16f), actual = actual[1])
    }

    @Test
    fun `train=勾配を伝播`() {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D1>

        assertEquals(expected = IOType.d1(0f, 12f, 24f), actual = actual[0])
        assertEquals(expected = IOType.d1(0f, 16f, 32f), actual = actual[1])
    }
}
