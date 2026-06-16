@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.converter.linear
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertContentEquals

class LinearD3Test {
    @Test
    fun `encode=Batchに変換`() = networkScopeTestRule {
        val target = LinearD3(outputX = 3, outputY = 3, outputZ = 3)
        val input = listOf(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })

        val actual = with(target) { encode(input) }

        val expected = Batch.of(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })
        assertContentEquals(expected = expected, actual = actual)
    }

    @Test
    fun `decode=Listに変換`() = networkScopeTestRule {
        val target = LinearD3(outputX = 3, outputY = 3, outputZ = 3)
        val input = Batch.of(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })

        val actual = with(target) { decode(input) }

        val expected = listOf(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })
        assertContentEquals(expected = expected, actual = actual)
    }
}
