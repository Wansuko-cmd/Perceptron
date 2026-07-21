@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.reshape.token

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import kotlin.test.Test

class TokenEmbeddingD1ToD2Test {
    val target = TokenEmbeddingD1ToD2(
        inputI = 3,
        outputJ = 5,
        vocabSize = 5,
        optimizer = Sgd(Scheduler.Fix(0.1f)).d2(5, 5),
        weight = IOType.d2(5, 5) { i, j -> i * 2f + j },
    )
    val input
        get() = Batch.of(
            IOType.d1(3) { it % 5f },
            IOType.d1(3) { -it % 4f + 4 },
        )

    @Test
    fun `expected=単語IDを重みの値に置換`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 1f, 2f, 3f, 4f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(2f, 3f, 4f, 5f, 6f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(4f, 5f, 6f, 7f, 8f), actual = actual[0][2])

        assertContentEquals(expected = IOType.d1(8f, 9f, 10f, 11f, 12f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(6f, 7f, 8f, 9f, 10f), actual = actual[1][1])
        assertContentEquals(expected = IOType.d1(4f, 5f, 6f, 7f, 8f), actual = actual[1][2])
    }

    @Test
    fun `train=重みを更新する`() = networkScopeTestRule {
        with(target) { _train(input = input, env = GraphEnv(), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, env = GraphEnv()) } as Batch<IOType.D2>

        assertContentEquals(expected = IOType.d1(0f, 0.95f, 1.9f, 2.85f, 3.8f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(1.9f, 2.85f, 3.8f, 4.75f, 5.7f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(3.6f, 4.5f, 5.4f, 6.3f, 7.2f), actual = actual[0][2])

        assertContentEquals(expected = IOType.d1(7.6f, 8.55f, 9.5f, 10.45f, 11.4f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(5.7f, 6.65f, 7.6f, 8.55f, 9.5f), actual = actual[1][1])
        assertContentEquals(expected = IOType.d1(3.6f, 4.5f, 5.4f, 6.3f, 7.2f), actual = actual[1][2])
    }
}
