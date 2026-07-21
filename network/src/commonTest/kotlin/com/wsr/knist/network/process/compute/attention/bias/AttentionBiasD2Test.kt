@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.attention.bias

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.core.get
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.GraphId
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.math.abs
import kotlin.math.pow
import kotlin.test.Test

class AttentionBiasD2Test {
    // scaled: (heads=2, seq=3, seq=3) のスコア行列。バイアスの影響だけを見たいので0で初期化する。
    private val scaled get() = Batch.of(IOType.d3(2, 3, 3) { _, _, _ -> 0f })
    private val dummyEnv get() = GraphEnv()

    @Test
    fun `Causal=queryより後ろのkey位置に-1e9を全ヘッド・全クエリ共通で加える`() = networkScopeTestRule {
        val bias = AttentionBiasD2.Causal(inputI = 3)

        val actual = with(bias) { forward(scaled, dummyEnv) }

        for (head in 0..1) {
            assertContentEquals(expected = IOType.d1(0f, -1e9f, -1e9f), actual = actual[0][head][0])
            assertContentEquals(expected = IOType.d1(0f, 0f, -1e9f), actual = actual[0][head][1])
            assertContentEquals(expected = IOType.d1(0f, 0f, 0f), actual = actual[0][head][2])
        }
    }

    @Test
    fun `Mask=指定した値のトークンに対応するkey位置に-1e9を全ヘッド・全クエリ共通で加える`() = networkScopeTestRule {
        val nodeId = GraphId()
        val bias = AttentionBiasD2.Mask(nodeId = nodeId, value = 0f)
        // 3番目のトークンだけpadding(0)扱いになる
        val env = GraphEnv()
        env[nodeId] = Batch.of(IOType.d1(1f, 1f, 0f))

        val actual = with(bias) { forward(scaled, env) }

        for (head in 0..1) {
            for (query in 0..2) {
                assertContentEquals(expected = IOType.d1(0f, 0f, -1e9f), actual = actual[0][head][query])
            }
        }
    }

    @Test
    fun `ALiBi=ヘッドごとに異なる傾きで距離に比例したバイアスを加える`() = networkScopeTestRule {
        val bias = AttentionBiasD2.ALiBi(numOfHeads = 2, inputI = 3)

        val actual = with(bias) { forward(scaled, dummyEnv) }

        fun slope(head: Int): Float = 2f.pow(-8f * (head + 1) / 2)
        fun expected(head: Int, query: Int, key: Int): Float = -slope(head) * abs(query - key).toFloat()

        for (head in 0..1) {
            for (query in 0..2) {
                assertContentEquals(
                    expected = IOType.d1(
                        expected(head, query, 0),
                        expected(head, query, 1),
                        expected(head, query, 2),
                    ),
                    actual = actual[0][head][query],
                    absoluteTolerance = 1e-4f,
                )
            }
        }
    }
}
