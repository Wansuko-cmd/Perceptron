@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.join.max

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class MaxD3Test {
    // 3番目の要素はa/bが同値タイ
    private val a = Batch.of(IOType.d3(IOType.d2(IOType.d1(1f, 5f, 3f))))
    private val b = Batch.of(IOType.d3(IOType.d2(IOType.d1(4f, 2f, 3f))))
    private val target get() = MaxD3(outputI = 1, outputJ = 1, outputK = 3)

    @Test
    fun `expect=要素ごとに大きい方の値を選ぶ`() = networkScopeTestRule {
        val actual = with(target) {
            _expect(inputs = listOf(a, b) as List<Batch<IOType>>, env = GraphEnv())
        } as Batch<IOType.D3>

        assertContentEquals(expected = Batch.of(IOType.d3(IOType.d2(IOType.d1(4f, 5f, 3f)))), actual = actual)
    }

    @Test
    fun `train=勝った入力にだけdeltaを流し負けた入力には0を返す_同値タイは両方に重複配分される`() = networkScopeTestRule {
        val actual = with(target) {
            _train(inputs = listOf(a, b) as List<Batch<IOType>>, env = GraphEnv(), calcDelta = { it })
        }

        assertContentEquals(
            expected = Batch.of(IOType.d3(IOType.d2(IOType.d1(0f, 5f, 3f)))),
            actual = actual[0] as Batch<IOType.D3>,
        )
        assertContentEquals(
            expected = Batch.of(IOType.d3(IOType.d2(IOType.d1(4f, 0f, 3f)))),
            actual = actual[1] as Batch<IOType.D3>,
        )
    }
}
