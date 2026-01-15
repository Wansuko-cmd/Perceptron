@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.network.process.compute.position

import com.wsr.batch.Batch
import com.wsr.batch.batchOf
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.get
import com.wsr.network.assertEquals
import com.wsr.network.networkTestRule
import com.wsr.network.process.Context
import kotlin.test.Test
import kotlin.test.assertEquals

class RoPED2Test {
    val target get() = RoPED2(outputX = 2, outputY = 4, waveLength = 100f)
    val input
        get() = batchOf(
            IOType.d2(
                IOType.d1(4) { it.toFloat() },
                IOType.d1(4) { it * 2f },
            ),
            IOType.d2(
                IOType.d1(4) { 10f % (it + 1) },
                IOType.d1(4) { it * 0.3f },
            ),
        )

    @Test
    fun `expect=RoPEアルゴリズムを元に位置情報埋め込み`() = networkTestRule {
        val actual = target._expect(input = input, context = Context(input)) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(0f, 1f, 2f, 3f), actual = actual[0][0])
        assertEquals(
            expected = IOType.d1(-1.6829f, 1.0806f, 3.3810f, 6.3693f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(expected = IOType.d1(0f, 0f, 1f, 2f), actual = actual[1][0])
        assertEquals(
            expected = IOType.d1(-0.2524f, 0.1620f, 0.5071f, 0.9554f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=勾配を伝播`() = networkTestRule {
        val actual = target._train(input = input, context = Context(input), calcDelta = { it }) as Batch<IOType.D2>

        assertEquals(expected = IOType.d1(0f, 1f, 2f, 3f), actual = actual[0][0])
        assertEquals(
            expected = IOType.d1(-1.8185f, -0.8322f, 2.7282f, 6.6750f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )

        assertEquals(expected = IOType.d1(0f, 0f, 1f, 2f), actual = actual[1][0])
        assertEquals(
            expected = IOType.d1(-0.2727f, -0.1248f, 0.4092f, 1.0012f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
    }
}
