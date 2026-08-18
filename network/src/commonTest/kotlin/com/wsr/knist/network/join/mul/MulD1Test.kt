@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.join.mul

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class MulD1Test {
    private val a = Batch.of(IOType.d1(2f, 3f))
    private val b = Batch.of(IOType.d1(4f, 5f))
    private val c = Batch.of(IOType.d1(6f, 7f))
    private val target get() = MulD1(outputI = 2)

    @Test
    fun `expect=要素積を計算する`() = networkScopeTestRule {
        val actual = with(target) {
            _expect(inputs = listOf(a, b, c) as List<Batch<IOType>>, env = GraphEnv())
        } as Batch<IOType.D1>

        assertContentEquals(expected = Batch.of(IOType.d1(48f, 105f)), actual = actual)
    }

    @Test
    fun `train=deltaを自分以外の入力の積で分配する`() = networkScopeTestRule {
        val actual = with(target) {
            _train(inputs = listOf(a, b, c) as List<Batch<IOType>>, env = GraphEnv(), calcDelta = { it })
        }

        // output = a*b*c = (48, 105)
        assertContentEquals(expected = Batch.of(IOType.d1(1152f, 3675f)), actual = actual[0] as Batch<IOType.D1>)
        assertContentEquals(expected = Batch.of(IOType.d1(576f, 2205f)), actual = actual[1] as Batch<IOType.D1>)
        assertContentEquals(expected = Batch.of(IOType.d1(384f, 1575f)), actual = actual[2] as Batch<IOType.D1>)
    }

    @Test
    fun `train=入力に0が含まれてもゼロ除算にならず正しく計算される`() = networkScopeTestRule {
        val zeroA = Batch.of(IOType.d1(0f, 3f))
        val zeroB = Batch.of(IOType.d1(4f, 0f))

        val actual = with(target) {
            _train(inputs = listOf(zeroA, zeroB) as List<Batch<IOType>>, env = GraphEnv(), calcDelta = { it })
        }

        assertContentEquals(expected = Batch.of(IOType.d1(0f, 0f)), actual = actual[0] as Batch<IOType.D1>)
        assertContentEquals(expected = Batch.of(IOType.d1(0f, 0f)), actual = actual[1] as Batch<IOType.D1>)
    }
}
