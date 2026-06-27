@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.converter.list
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test

class LinearD3Test {
    @Test
    fun `encode=Batchに変換`() = networkTestRule {
        val target = LinearD3(outputI = 3, outputJ = 3, outputK = 3)
        val input = listOf(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })

        val actual = target.encode(input)

        val expected = Batch.of(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })
        assertContentEquals(expected = expected, actual = actual)
    }

    @Test
    fun `decode=Listに変換`() = networkTestRule {
        val target = LinearD3(outputI = 3, outputJ = 3, outputK = 3)
        val input = Batch.of(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })

        val actual = target.decode(input)

        val expected = listOf(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })
        assertContentEquals(expected = expected, actual = actual)
    }
}
