@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.converter.list
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test

class LinearD1Test {
    @Test
    fun `encode=Batchに変換`() = networkTestRule {
        val target = ListD1(outputI = 3)
        val input = listOf(IOType.d1(3) { it.toFloat() })

        val actual = target.encode(input)

        assertContentEquals(expected = Batch.of(IOType.d1(3) { it.toFloat() }), actual = actual)
    }

    @Test
    fun `decode=Listに変換`() = networkTestRule {
        val target = ListD1(outputI = 3)
        val input = Batch.of(IOType.d1(3) { it.toFloat() })

        val actual = target.decode(input)

        assertContentEquals(expected = listOf(IOType.d1(3) { it.toFloat() }), actual = actual)
    }
}
