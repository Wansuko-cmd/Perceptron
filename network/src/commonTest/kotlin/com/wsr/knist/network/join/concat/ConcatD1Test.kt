@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.join.concat

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class ConcatD1Test {
    private val a = Batch.of(IOType.d1(1f, 2f), IOType.d1(3f, 4f))
    private val b = Batch.of(IOType.d1(10f, 20f, 30f), IOType.d1(40f, 50f, 60f))
    private val c = Batch.of(IOType.d1(100f), IOType.d1(200f))
    private val target get() = ConcatD1(outputI = 6)

    @Test
    fun `expect=複数入力を連結する`() = networkScopeTestRule {
        val actual = with(target) {
            _expect(inputs = listOf(a, b, c) as List<Batch<IOType>>, env = GraphEnv())
        } as Batch<IOType.D1>

        assertContentEquals(
            expected = Batch.of(
                IOType.d1(1f, 2f, 10f, 20f, 30f, 100f),
                IOType.d1(3f, 4f, 40f, 50f, 60f, 200f),
            ),
            actual = actual,
        )
    }

    @Test
    fun `train=中間の入力を含めてdeltaを連結順に分配する`() = networkScopeTestRule {
        val actual = with(target) {
            _train(inputs = listOf(a, b, c) as List<Batch<IOType>>, env = GraphEnv(), calcDelta = { it })
        }

        assertContentEquals(expected = a, actual = actual[0] as Batch<IOType.D1>)
        assertContentEquals(expected = b, actual = actual[1] as Batch<IOType.D1>)
        assertContentEquals(expected = c, actual = actual[2] as Batch<IOType.D1>)
    }
}
