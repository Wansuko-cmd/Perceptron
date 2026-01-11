@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.reshape.token

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.NetworkTestRule
import com.wsr.optimizer.Scheduler
import com.wsr.optimizer.sgd.Sgd
import com.wsr.process.Context
import com.wsr.process.reshape.token.TokenEmbeddingD1ToD2
import org.junit.Rule
import kotlin.test.Test
import kotlin.test.assertEquals

class TokenEmbeddingD1ToD2Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target = TokenEmbeddingD1ToD2(
        outputX = 3,
        outputY = 5,
        vocabSize = 5,
        optimizer = Sgd(Scheduler.Fix(0.1f)).d2(5, 5),
        weight = IOType.d2(5, 5) { i, j -> i * 2f + j },
    )
    val input
        get() = batchOf(
            IOType.d1(5) { it % 5f },
            IOType.d1(5) { -it % 4f + 4 },
        )

    @Test
    fun `expected=単語IDを重みの値に置換`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(0f, 1f, 2f, 3f, 4f), actual = actual[0][0])
        assertEquals(expected = IOType.d1(2f, 3f, 4f, 5f, 6f), actual = actual[0][1])
        assertEquals(expected = IOType.d1(4f, 5f, 6f, 7f, 8f), actual = actual[0][2])

        assertEquals(expected = IOType.d1(8f, 9f, 10f, 11f, 12f), actual = actual[1][0])
        assertEquals(expected = IOType.d1(6f, 7f, 8f, 9f, 10f), actual = actual[1][1])
        assertEquals(expected = IOType.d1(4f, 5f, 6f, 7f, 8f), actual = actual[1][2])
    }

    @Test
    fun `train=重みを更新する`() {
        target._train(input = input, context = Context(input), calcDelta = { it })
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(0f, 0.95f, 1.9f, 2.85f, 3.8f), actual = actual[0][0])
        assertEquals(expected = IOType.d1(1.9f, 2.85f, 3.8f, 4.75f, 5.7f), actual = actual[0][1])
        assertEquals(expected = IOType.d1(3.6f, 4.5f, 5.4f, 6.3f, 7.2f), actual = actual[0][2])

        assertEquals(expected = IOType.d1(7.6f, 8.55f, 9.5f, 10.45f, 11.4f), actual = actual[1][0])
        assertEquals(expected = IOType.d1(5.7f, 6.65f, 7.6f, 8.55f, 9.5f), actual = actual[1][1])
        assertEquals(expected = IOType.d1(3.6f, 4.5f, 5.4f, 6.3f, 7.2f), actual = actual[1][2])
    }
}
