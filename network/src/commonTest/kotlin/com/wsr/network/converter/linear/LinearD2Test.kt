@file:Suppress("NonAsciiCharacters")

package com.wsr.network.converter.linear

import com.wsr.batch.batchOf
import com.wsr.core.IOType
import com.wsr.core.d2
import com.wsr.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class LinearD2Test {
    @Test
    fun `encode=Batchに変換`() = networkTestRule {
        val target = LinearD2(outputX = 3, outputY = 3)
        val input = listOf(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })

        val actual = target.encode(input)

        val expected = batchOf(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })
        assertEquals(expected = expected, actual = actual)
    }

    @Test
    fun `decode=Listに変換`() = networkTestRule {
        val target = LinearD2(outputX = 3, outputY = 3)
        val input = batchOf(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })

        val actual = target.decode(input)

        val expected = listOf(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })
        assertEquals(expected = expected, actual = actual)
    }
}
