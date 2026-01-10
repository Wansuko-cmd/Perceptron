@file:Suppress("NonAsciiCharacters")

package com.wsr.network.converter.linear

import com.wsr.batch.batchOf
import com.wsr.converter.linear.LinearD1
import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.network.NetworkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule

class LinearD1Test {
    @get:Rule
    val networkTestRule = NetworkTestRule()

    @Test
    fun `encode=Batchに変換`() {
        val target = LinearD1(outputSize = 3)
        val input = listOf(IOType.d1(3) { it.toFloat() })

        val actual = target.encode(input)

        assertEquals(expected = batchOf(IOType.d1(3) { it.toFloat() }), actual = actual)
    }

    @Test
    fun `decode=Listに変換`() {
        val target = LinearD1(outputSize = 3)
        val input = batchOf(IOType.d1(3) { it.toFloat() })

        val actual = target.decode(input)

        assertEquals(expected = listOf(IOType.d1(3) { it.toFloat() }), actual = actual)
    }
}
