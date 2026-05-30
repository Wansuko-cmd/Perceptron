@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.converter.linear

import com.wsr.knist.batch.batchOf
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d2
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertContentEquals

class LinearD2Test {
    @Test
    fun `encode=Batchに変換`() = networkTestRule {
        val target = LinearD2(outputX = 3, outputY = 3)
        val input = listOf(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })

        val actual = target.encode(input)

        val expected = batchOf(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })
        assertContentEquals(expected = expected, actual = actual)
    }

    @Test
    fun `decode=Listに変換`() = networkTestRule {
        val target = LinearD2(outputX = 3, outputY = 3)
        val input = batchOf(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })

        val actual = target.decode(input)

        val expected = listOf(IOType.d2(2, 2) { i, j -> i.toFloat() + j.toFloat() })
        assertContentEquals(expected = expected, actual = actual)
    }
}
