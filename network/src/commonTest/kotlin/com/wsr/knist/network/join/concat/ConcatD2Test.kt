@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.join.concat

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class ConcatD2Test {
    private val a = Batch.of(IOType.d2(IOType.d1(1f, 2f)))
    private val b = Batch.of(IOType.d2(IOType.d1(3f, 4f), IOType.d1(5f, 6f)))
    private val c = Batch.of(IOType.d2(IOType.d1(7f, 8f)))

    private val d = Batch.of(IOType.d2(IOType.d1(1f), IOType.d1(2f)))
    private val e = Batch.of(IOType.d2(IOType.d1(3f, 4f), IOType.d1(5f, 6f)))
    private val f = Batch.of(IOType.d2(IOType.d1(7f), IOType.d1(8f)))

    @Test
    fun `expect_axis0=行方向に連結する`() = networkScopeTestRule {
        val target = ConcatD2(outputI = 4, outputJ = 2, axis = 0)
        val actual = with(target) {
            _expect(inputs = listOf(a, b, c) as List<Batch<IOType>>, env = GraphEnv())
        } as Batch<IOType.D2>

        assertContentEquals(
            expected = Batch.of(
                IOType.d2(IOType.d1(1f, 2f), IOType.d1(3f, 4f), IOType.d1(5f, 6f), IOType.d1(7f, 8f)),
            ),
            actual = actual,
        )
    }

    @Test
    fun `train_axis0=中間の入力を含めてdeltaを行方向に分配する`() = networkScopeTestRule {
        val target = ConcatD2(outputI = 4, outputJ = 2, axis = 0)
        val actual = with(target) {
            _train(inputs = listOf(a, b, c) as List<Batch<IOType>>, env = GraphEnv(), calcDelta = { it })
        }

        assertContentEquals(expected = a, actual = actual[0] as Batch<IOType.D2>)
        assertContentEquals(expected = b, actual = actual[1] as Batch<IOType.D2>)
        assertContentEquals(expected = c, actual = actual[2] as Batch<IOType.D2>)
    }

    @Test
    fun `expect_axis1=列方向に連結する`() = networkScopeTestRule {
        val target = ConcatD2(outputI = 2, outputJ = 4, axis = 1)
        val actual = with(target) {
            _expect(inputs = listOf(d, e, f) as List<Batch<IOType>>, env = GraphEnv())
        } as Batch<IOType.D2>

        assertContentEquals(
            expected = Batch.of(IOType.d2(IOType.d1(1f, 3f, 4f, 7f), IOType.d1(2f, 5f, 6f, 8f))),
            actual = actual,
        )
    }

    @Test
    fun `train_axis1=中間の入力を含めてdeltaを列方向に分配する`() = networkScopeTestRule {
        val target = ConcatD2(outputI = 2, outputJ = 4, axis = 1)
        val actual = with(target) {
            _train(inputs = listOf(d, e, f) as List<Batch<IOType>>, env = GraphEnv(), calcDelta = { it })
        }

        assertContentEquals(expected = d, actual = actual[0] as Batch<IOType.D2>)
        assertContentEquals(expected = e, actual = actual[1] as Batch<IOType.D2>)
        assertContentEquals(expected = f, actual = actual[2] as Batch<IOType.D2>)
    }
}
