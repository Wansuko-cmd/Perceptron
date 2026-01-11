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
import com.wsr.process.Context
import com.wsr.process.compute.position.PositionEncodeD2
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class PositionEncodeD2Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    val target get() = PositionEncodeD2(outputX = 2, outputY = 3, waveLength = 100f)
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
    fun `expect=位置情報埋め込み`() {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(0f, 2f, 2f), actual = actual[0][0])
        assertEquals(
            expected = IOType.d1(0.8414f, 2.5403f, 4.0463f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(expected = IOType.d1(0f, 1f, 1f), actual = actual[1][0])
        assertEquals(
            expected = IOType.d1(0.8414f, 0.8403f, 0.6463f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=勾配を伝播`() {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(0f, 2f, 2f), actual = actual[0][0])
        assertEquals(
            expected = IOType.d1(0.8414f, 2.5403f, 4.0463f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(expected = IOType.d1(0f, 1f, 1f), actual = actual[1][0])
        assertEquals(
            expected = IOType.d1(0.8414f, 0.8403f, 0.6463f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
