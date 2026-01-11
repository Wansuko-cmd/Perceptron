@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.position

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import com.wsr.network.assertEquals
import com.wsr.optimizer.Scheduler
import com.wsr.optimizer.sgd.Sgd
import com.wsr.process.Context
import com.wsr.process.compute.position.PositionEmbeddingD2
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class PositionEmbeddingD2Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target
        get() = PositionEmbeddingD2(
            outputX = 2,
            outputY = 3,
            optimizer = Sgd(scheduler = Scheduler.Fix(rate = 0.01f)).d2(2, 3),
            weight = IOType.d2(2, 3) { i, j -> i * 2f + j },
        )
    val input
        get() = batchOf(
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
    fun `expect=学習型位置情報埋め込み`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(0f, 2f, 4f), actual = actual[0][0])
        assertEquals(expected = IOType.d1(2f, 5f, 8f), actual = actual[0][1])

        assertEquals(expected = IOType.d1(0f, 1f, 3f), actual = actual[1][0])
        assertEquals(expected = IOType.d1(2f, 3.3f, 4.6f), actual = actual[1][1])
    }

    @Test
    fun `train=勾配を伝播`() {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(0f, 2f, 4f), actual = actual[0][0])
        assertEquals(expected = IOType.d1(2f, 5f, 8f), actual = actual[0][1])

        assertEquals(expected = IOType.d1(0f, 1f, 3f), actual = actual[1][0])
        assertEquals(expected = IOType.d1(2f, 3.3f, 4.6f), actual = actual[1][1])
    }

    @Test
    fun `train=重みを更新`() {
        val target = target

        target._train(input = input, context = Context(input), calcDelta = { it })
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(
            expected = IOType.d1(0f, 1.985f, 3.9650f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(1.98f, 4.9585f, 7.9370f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(
            expected = IOType.d1(0f, 0.985f, 2.9650f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertEquals(
            expected = IOType.d1(1.98f, 3.2584f, 4.537f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
