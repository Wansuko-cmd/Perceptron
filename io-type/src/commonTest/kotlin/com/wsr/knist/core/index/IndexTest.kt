@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.core.index

import com.wsr.knist.assertContentEquals
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.index.gather
import com.wsr.knist.core.index.scatterAdd
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class IndexTest {
    @Test
    fun `gather=インデックスに対応する行を収集`() = ioTypeTestRule {
        val d1 = IOType.d1(listOf(0f, 1f, 0f))
        val lookup = IOType.d2(listOf(2, 2), listOf(10f, 11f, 20f, 21f))
        val actual = d1.gather(lookup)
        assertContentEquals(
            expected = IOType.d2(listOf(3, 2), listOf(10f, 11f, 20f, 21f, 10f, 11f)),
            actual = actual,
        )
    }

    @Test
    fun `scatterAdd=インデックスに対応する行を集計`() = ioTypeTestRule {
        val d2 = IOType.d2(listOf(3, 2), listOf(1f, 2f, 3f, 4f, 5f, 6f))
        val d1 = IOType.d1(listOf(0f, 1f, 0f))
        val actual = d2.scatterAdd(d1, n = 2)
        assertContentEquals(
            expected = IOType.d2(listOf(2, 2), listOf(6f, 8f, 3f, 4f)),
            actual = actual,
        )
    }
}
