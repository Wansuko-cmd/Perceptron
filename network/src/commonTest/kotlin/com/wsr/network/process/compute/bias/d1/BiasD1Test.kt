@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.bias.d1

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
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class BiasD1Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target
        get() = BiasD1(
            outputSize = 3,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d1(3),
            weight = IOType.d1(3) { it.toFloat() },
        )

    val input get() = batchOf(
        IOType.d1(3) { it * 2f },
        IOType.d1(3) { it * 3f },
    )

    @Test
    fun `expect=バイアス項`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D1>

        assertEquals(expected = IOType.d1(0f, 3f, 6f), actual = actual[0])
        assertEquals(expected = IOType.d1(0f, 4f, 8f), actual = actual[1])
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D1>

        assertEquals(expected = IOType.d1(0f, 3f, 6f), actual = actual[0])
        assertEquals(expected = IOType.d1(0f, 4f, 8f), actual = actual[1])
    }

    @Test
    fun `train=重みを更新する`() {
        val target = target

        target._train(input = input, context = Context(input), calcDelta = { it })
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D1>

        assertEquals(expected = IOType.d1(0f, 2.965f, 5.93f), actual = actual[0])
        assertEquals(expected = IOType.d1(0f, 3.965f, 7.93f), actual = actual[1])
    }
}
