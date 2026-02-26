@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.scale.d1

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.network.assertContentEquals
import com.wsr.network.networkTestRule
import com.wsr.network.optimizer.Scheduler
import com.wsr.network.optimizer.sgd.Sgd
import com.wsr.network.process.Context
import kotlin.test.Test
import kotlin.test.assertContentEquals

class ScaleD1Test {
    val target
        get() = ScaleD1(
            outputSize = 3,
            optimizer = Sgd(Scheduler.Fix(0.01f)).d1(3),
            weight = IOType.d1(3) { it.toFloat() },
        )

    val input get() = batchOf(
        IOType.d1(3) { it * 2f },
        IOType.d1(3) { it * 3f },
    )

    @Test
    fun `expect=スケール項`() = networkTestRule {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d1(0f, 2f, 8f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(0f, 3f, 12f), actual = actual[1])
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() = networkTestRule {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d1(0f, 2f, 16f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(0f, 3f, 24f), actual = actual[1])
    }

    @Test
    fun `train=重みを更新する`() = networkTestRule {
        val target = target

        target._train(input = input, context = Context(input), calcDelta = { it })
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D1>

        assertContentEquals(expected = IOType.d1(0f, 1.87f, 5.92f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(0f, 2.805f, 8.88f), actual = actual[1])
    }
}
