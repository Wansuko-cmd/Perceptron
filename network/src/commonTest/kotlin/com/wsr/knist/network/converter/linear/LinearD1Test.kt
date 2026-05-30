@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.converter.linear

import com.wsr.knist.batch.batchOf
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertContentEquals

class LinearD1Test {
    @Test
    fun `encode=Batchに変換`() = networkTestRule {
        val target = LinearD1(outputSize = 3)
        val input = listOf(IOType.d1(3) { it.toFloat() })

        val actual = target.encode(input)

        assertContentEquals(expected = batchOf(IOType.d1(3) { it.toFloat() }), actual = actual)
    }

    @Test
    fun `decode=Listに変換`() = networkTestRule {
        val target = LinearD1(outputSize = 3)
        val input = batchOf(IOType.d1(3) { it.toFloat() })

        val actual = target.decode(input)

        assertContentEquals(expected = listOf(IOType.d1(3) { it.toFloat() }), actual = actual)
    }
}
