@file:Suppress("NonAsciiCharacters")

package com.wsr.network.converter.linear

import com.wsr.batch.batchOf
import com.wsr.core.IOType
import com.wsr.core.d3
import com.wsr.network.assertContentEquals
import com.wsr.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertContentEquals

class LinearD3Test {
    @Test
    fun `encode=Batchに変換`() = networkTestRule {
        val target = LinearD3(outputX = 3, outputY = 3, outputZ = 3)
        val input = listOf(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })

        val actual = target.encode(input)

        val expected = batchOf(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })
        assertContentEquals(expected = expected, actual = actual)
    }

    @Test
    fun `decode=Listに変換`() = networkTestRule {
        val target = LinearD3(outputX = 3, outputY = 3, outputZ = 3)
        val input = batchOf(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })

        val actual = target.decode(input)

        val expected = listOf(IOType.d3(2, 2, 2) { i, j, k -> i.toFloat() + j.toFloat() + k.toFloat() })
        assertContentEquals(expected = expected, actual = actual)
    }
}
