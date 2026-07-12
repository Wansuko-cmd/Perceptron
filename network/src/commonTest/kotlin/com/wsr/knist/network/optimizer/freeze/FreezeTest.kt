@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network.optimizer.freeze

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class FreezeTest {
    @Test
    fun `D1_adapt=重みを更新しない`() = networkScopeTestRule {
        val target = Freeze.d1(3)
        val weight = IOType.d1(1f, 2f, 3f)
        val dw = IOType.d1(10f, 20f, 30f)

        val actual = target.adapt(weight, dw)

        assertContentEquals(expected = IOType.d1(1f, 2f, 3f), actual = actual)
    }

    @Test
    fun `D2_adapt=重みを更新しない`() = networkScopeTestRule {
        val target = Freeze.d2(i = 2, j = 2)
        val weight = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val dw = IOType.d2(2, 2) { _, _ -> 10f }

        val actual = target.adapt(weight, dw)

        assertContentEquals(expected = IOType.d1(0f, 1f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(2f, 3f), actual = actual[1])
    }

    @Test
    fun `D3_adapt=重みを更新しない`() = networkScopeTestRule {
        val target = Freeze.d3(i = 2, j = 2, k = 2)
        val weight = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k }
        val dw = IOType.d3(2, 2, 2) { _, _, _ -> 10f }

        val actual = target.adapt(weight, dw)

        assertContentEquals(expected = IOType.d1(0f, 1f), actual = actual[0][0])
        assertContentEquals(expected = IOType.d1(2f, 3f), actual = actual[0][1])
        assertContentEquals(expected = IOType.d1(4f, 5f), actual = actual[1][0])
        assertContentEquals(expected = IOType.d1(6f, 7f), actual = actual[1][1])
    }

    @Test
    fun `D4_adapt=重みを更新しない`() = networkScopeTestRule {
        val target = Freeze.d4(i = 2, j = 2, k = 2, l = 2)
        val weight = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }
        val dw = IOType.d4(2, 2, 2, 2) { _, _, _, _ -> 10f }

        val actual = target.adapt(weight, dw)

        assertContentEquals(expected = IOType.d1(0f, 1f), actual = actual[0][0][0])
        assertContentEquals(expected = IOType.d1(2f, 3f), actual = actual[0][0][1])
        assertContentEquals(expected = IOType.d1(4f, 5f), actual = actual[0][1][0])
        assertContentEquals(expected = IOType.d1(6f, 7f), actual = actual[0][1][1])
        assertContentEquals(expected = IOType.d1(8f, 9f), actual = actual[1][0][0])
        assertContentEquals(expected = IOType.d1(10f, 11f), actual = actual[1][0][1])
        assertContentEquals(expected = IOType.d1(12f, 13f), actual = actual[1][1][0])
        assertContentEquals(expected = IOType.d1(14f, 15f), actual = actual[1][1][1])
    }

    @Test
    fun `D1_adapt=Batchの勾配でも重みを更新しない`() = networkScopeTestRule {
        val target = Freeze.d1(3)
        val weight = IOType.d1(1f, 2f, 3f)
        val dw = Batch.of(
            IOType.d1(10f, 20f, 30f),
            IOType.d1(40f, 50f, 60f),
        )

        val actual = target.adapt(weight, dw)

        assertContentEquals(expected = IOType.d1(1f, 2f, 3f), actual = actual)
    }

    @Test
    fun `D2_adapt=Batchの勾配でも重みを更新しない`() = networkScopeTestRule {
        val target = Freeze.d2(i = 2, j = 2)
        val weight = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val dw = Batch.of(
            IOType.d2(2, 2) { _, _ -> 10f },
            IOType.d2(2, 2) { _, _ -> 20f },
        )

        val actual = target.adapt(weight, dw)

        assertContentEquals(expected = IOType.d1(0f, 1f), actual = actual[0])
        assertContentEquals(expected = IOType.d1(2f, 3f), actual = actual[1])
    }
}
