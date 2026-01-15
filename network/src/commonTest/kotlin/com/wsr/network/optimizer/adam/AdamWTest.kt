@file:Suppress("NonAsciiCharacters")

package com.wsr.network.optimizer.adam

import com.wsr.core.IOType
import com.wsr.core.d1
import com.wsr.core.d2
import com.wsr.core.d3
import com.wsr.core.d4
import com.wsr.core.get
import com.wsr.network.networkTestRule
import com.wsr.network.optimizer.Scheduler
import kotlin.test.Test
import kotlin.test.assertEquals

class AdamWTest {
    @Test
    fun `D1_adapt=AdamWアルゴリズムで最適化`() = networkTestRule {
        val target = AdamW(scheduler = Scheduler.Fix(1f)).d1(3)
        val weight = IOType.d1(1f, 2f, 3f)
        val dw = IOType.d1(1f, 2f, 3f)

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = -0.0099f, actual = actual1[0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9800f, actual = actual1[1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.9700f, actual = actual1[2], absoluteTolerance = 1e-4f)

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = -0.0099f, actual = actual2[0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9800f, actual = actual2[1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.9700f, actual = actual2[2], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D2_adapt=AdamWアルゴリズムで最適化`() = networkTestRule {
        val target = AdamW(scheduler = Scheduler.Fix(1f)).d2(i = 2, j = 2)
        val weight = IOType.d2(2, 2) { i, j -> i * 2f + j }
        val dw = IOType.d2(2, 2) { i, j -> i * 2f + j }

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual1[0][0])
        assertEquals(expected = -0.0099f, actual = actual1[0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9800f, actual = actual1[1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.9700f, actual = actual1[1][1], absoluteTolerance = 1e-4f)

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual2[0][0])
        assertEquals(expected = -0.0099f, actual = actual2[0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9800f, actual = actual2[1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.9700f, actual = actual2[1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D3_adapt=AdamWアルゴリズムで最適化`() = networkTestRule {
        val target = AdamW(scheduler = Scheduler.Fix(1f)).d3(i = 2, j = 2, k = 2)
        val weight = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k }
        val dw = IOType.d3(2, 2, 2) { i, j, k -> i * 4f + j * 2f + k }

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual1[0][0][0])
        assertEquals(expected = -0.0099f, actual = actual1[0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9800f, actual = actual1[0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.9700f, actual = actual1[0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 2.9600f, actual = actual1[1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 3.9499f, actual = actual1[1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 4.9400f, actual = actual1[1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 5.9300f, actual = actual1[1][1][1], absoluteTolerance = 1e-4f)

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual2[0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = -0.0099f, actual = actual2[0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9800f, actual = actual2[0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.9700f, actual = actual2[0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 2.9600f, actual = actual2[1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 3.9500f, actual = actual2[1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 4.9400f, actual = actual2[1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 5.9300f, actual = actual2[1][1][1], absoluteTolerance = 1e-4f)
    }

    @Test
    fun `D4_adapt=AdamWアルゴリズムで最適化`() = networkTestRule {
        val target = AdamW(scheduler = Scheduler.Fix(1f)).d4(i = 2, j = 2, k = 2, l = 2)
        val weight = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }
        val dw = IOType.d4(2, 2, 2, 2) { i, j, k, l -> i * 8f + j * 4f + k * 2f + l }

        val actual1 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual1[0][0][0][0])
        assertEquals(expected = -0.0099f, actual = actual1[0][0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9800f, actual = actual1[0][0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.9700f, actual = actual1[0][0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 2.9600f, actual = actual1[0][1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 3.9499f, actual = actual1[0][1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 4.9400f, actual = actual1[0][1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 5.9300f, actual = actual1[0][1][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 6.9200f, actual = actual1[1][0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 7.9100f, actual = actual1[1][0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 8.9000f, actual = actual1[1][0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 9.8900f, actual = actual1[1][0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 10.8800f, actual = actual1[1][1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 11.8700f, actual = actual1[1][1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 12.8600f, actual = actual1[1][1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 13.8500f, actual = actual1[1][1][1][1], absoluteTolerance = 1e-4f)

        val actual2 = target.adapt(weight, dw)

        assertEquals(expected = 0f, actual = actual2[0][0][0][0])
        assertEquals(expected = -0.0099f, actual = actual2[0][0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 0.9800f, actual = actual2[0][0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 1.9700f, actual = actual2[0][0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 2.9600f, actual = actual2[0][1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 3.9500f, actual = actual2[0][1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 4.9400f, actual = actual2[0][1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 5.9300f, actual = actual2[0][1][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 6.9200f, actual = actual2[1][0][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 7.9100f, actual = actual2[1][0][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 8.9000f, actual = actual2[1][0][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 9.8900f, actual = actual2[1][0][1][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 10.8800f, actual = actual2[1][1][0][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 11.8700f, actual = actual2[1][1][0][1], absoluteTolerance = 1e-4f)
        assertEquals(expected = 12.8600f, actual = actual2[1][1][1][0], absoluteTolerance = 1e-4f)
        assertEquals(expected = 13.8500f, actual = actual2[1][1][1][1], absoluteTolerance = 1e-4f)
    }
}
