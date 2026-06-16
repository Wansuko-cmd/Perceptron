@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.converter.char
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.get
import com.wsr.knist.core.set
import com.wsr.knist.core.unwrap
import com.wsr.knist.network.networkTestRule
import kotlin.test.Test
import kotlin.test.assertEquals

class CharD1Test {
    @Test
    fun `encode=文字をone-hotベクトルに変換`() = networkTestRule {
        val target = CharD1()
        val input = listOf('a', 'b', 'c')

        val actual = target.encode(input)

        assertEquals(expected = CharD1.vocabSize, actual = actual[0].size)
        assertEquals(expected = 1f, actual = actual[0][1].unwrap())
        assertEquals(expected = 1f, actual = actual[1][2].unwrap())
        assertEquals(expected = 1f, actual = actual[2][3].unwrap())
    }

    @Test
    fun `decode=one-hotベクトルを文字に変換`() = networkTestRule {
        val target = CharD1()
        val input = Batch.of(
            IOType.d1(CharD1.vocabSize).also { it[1] = 1.0f },
            IOType.d1(CharD1.vocabSize).also { it[2] = 1.0f },
            IOType.d1(CharD1.vocabSize).also { it[3] = 1.0f },
        )

        val actual = target.decode(input)

        assertEquals(expected = 3, actual = actual.size)
        assertEquals(expected = 'a', actual = actual[0])
        assertEquals(expected = 'b', actual = actual[1])
        assertEquals(expected = 'c', actual = actual[2])
    }
}
