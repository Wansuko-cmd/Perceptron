@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.join.add

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class AddD1Test {
    private val a = Batch.of(IOType.d1(1f, 2f))
    private val b = Batch.of(IOType.d1(3f, 4f))
    private val c = Batch.of(IOType.d1(5f, 6f))
    private val target get() = AddD1(outputI = 2)

    @Test
    fun `expect=全入力を要素ごとに加算する`() = networkScopeTestRule {
        val actual = with(target) {
            _expect(inputs = listOf(a, b, c) as List<Batch<IOType>>, env = GraphEnv())
        } as Batch<IOType.D1>

        assertContentEquals(expected = Batch.of(IOType.d1(9f, 12f)), actual = actual)
    }

    @Test
    fun `train=deltaを全入力へそのまま分配する`() = networkScopeTestRule {
        val actual = with(target) {
            _train(inputs = listOf(a, b, c) as List<Batch<IOType>>, env = GraphEnv(), calcDelta = { it })
        }

        val expectedDelta = Batch.of(IOType.d1(9f, 12f))
        assertContentEquals(expected = expectedDelta, actual = actual[0] as Batch<IOType.D1>)
        assertContentEquals(expected = expectedDelta, actual = actual[1] as Batch<IOType.D1>)
        assertContentEquals(expected = expectedDelta, actual = actual[2] as Batch<IOType.D1>)
    }
}
