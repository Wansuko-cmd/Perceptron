@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.converter.linear
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test

class LinearD2Test {
    @Test
    fun `encode=Batchに変換`() = networkTestRule {
        val target = LinearD2(outputI = 3, outputJ = 3)
        val input = listOf(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })

        val actual = target.encode(input)

        val expected = Batch.of(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })
        assertContentEquals(expected = expected, actual = actual)
    }

    @Test
    fun `decode=Listに変換`() = networkTestRule {
        val target = LinearD2(outputI = 3, outputJ = 3)
        val input = Batch.of(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })

        val actual = target.decode(input)

        val expected = listOf(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })
        assertContentEquals(expected = expected, actual = actual)
    }
}
