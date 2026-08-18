@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.join.concat

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.network.GraphEnv
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import kotlin.test.Test

class ConcatD3Test {
    // axis=0（i方向）: 同じ j, k でプレーン数が異なる
    private val a0 = Batch.of(IOType.d3(IOType.d2(IOType.d1(1f, 2f), IOType.d1(3f, 4f))))
    private val b0 = Batch.of(
        IOType.d3(
            IOType.d2(IOType.d1(5f, 6f), IOType.d1(7f, 8f)),
            IOType.d2(IOType.d1(9f, 10f), IOType.d1(11f, 12f)),
        ),
    )
    private val c0 = Batch.of(IOType.d3(IOType.d2(IOType.d1(13f, 14f), IOType.d1(15f, 16f))))

    // axis=1（j方向）: 同じ i, k で行数が異なる
    private val a1 = Batch.of(IOType.d3(IOType.d2(IOType.d1(1f, 2f))))
    private val b1 = Batch.of(IOType.d3(IOType.d2(IOType.d1(3f, 4f), IOType.d1(5f, 6f))))
    private val c1 = Batch.of(IOType.d3(IOType.d2(IOType.d1(7f, 8f))))

    // axis=2（k方向）: 同じ i, j で列数が異なる
    private val a2 = Batch.of(IOType.d3(IOType.d2(IOType.d1(1f))))
    private val b2 = Batch.of(IOType.d3(IOType.d2(IOType.d1(2f, 3f))))
    private val c2 = Batch.of(IOType.d3(IOType.d2(IOType.d1(4f))))

    @Test
    fun `expect_axis0=プレーン方向に連結する`() = networkScopeTestRule {
        val target = ConcatD3(outputI = 4, outputJ = 2, outputK = 2, axis = 0)
        val actual = with(target) {
            _expect(inputs = listOf(a0, b0, c0) as List<Batch<IOType>>, env = GraphEnv())
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = Batch.of(
                IOType.d3(
                    IOType.d2(IOType.d1(1f, 2f), IOType.d1(3f, 4f)),
                    IOType.d2(IOType.d1(5f, 6f), IOType.d1(7f, 8f)),
                    IOType.d2(IOType.d1(9f, 10f), IOType.d1(11f, 12f)),
                    IOType.d2(IOType.d1(13f, 14f), IOType.d1(15f, 16f)),
                ),
            ),
            actual = actual,
        )
    }

    @Test
    fun `train_axis0=中間の入力を含めてdeltaをプレーン方向に分配する`() = networkScopeTestRule {
        val target = ConcatD3(outputI = 4, outputJ = 2, outputK = 2, axis = 0)
        val actual = with(target) {
            _train(inputs = listOf(a0, b0, c0) as List<Batch<IOType>>, env = GraphEnv(), calcDelta = { it })
        }

        assertContentEquals(expected = a0, actual = actual[0] as Batch<IOType.D3>)
        assertContentEquals(expected = b0, actual = actual[1] as Batch<IOType.D3>)
        assertContentEquals(expected = c0, actual = actual[2] as Batch<IOType.D3>)
    }

    @Test
    fun `expect_axis1=行方向に連結する`() = networkScopeTestRule {
        val target = ConcatD3(outputI = 1, outputJ = 4, outputK = 2, axis = 1)
        val actual = with(target) {
            _expect(inputs = listOf(a1, b1, c1) as List<Batch<IOType>>, env = GraphEnv())
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = Batch.of(
                IOType.d3(
                    IOType.d2(IOType.d1(1f, 2f), IOType.d1(3f, 4f), IOType.d1(5f, 6f), IOType.d1(7f, 8f)),
                ),
            ),
            actual = actual,
        )
    }

    @Test
    fun `train_axis1=中間の入力を含めてdeltaを行方向に分配する`() = networkScopeTestRule {
        val target = ConcatD3(outputI = 1, outputJ = 4, outputK = 2, axis = 1)
        val actual = with(target) {
            _train(inputs = listOf(a1, b1, c1) as List<Batch<IOType>>, env = GraphEnv(), calcDelta = { it })
        }

        assertContentEquals(expected = a1, actual = actual[0] as Batch<IOType.D3>)
        assertContentEquals(expected = b1, actual = actual[1] as Batch<IOType.D3>)
        assertContentEquals(expected = c1, actual = actual[2] as Batch<IOType.D3>)
    }

    @Test
    fun `expect_axis2=列方向に連結する`() = networkScopeTestRule {
        val target = ConcatD3(outputI = 1, outputJ = 1, outputK = 4, axis = 2)
        val actual = with(target) {
            _expect(inputs = listOf(a2, b2, c2) as List<Batch<IOType>>, env = GraphEnv())
        } as Batch<IOType.D3>

        assertContentEquals(
            expected = Batch.of(IOType.d3(IOType.d2(IOType.d1(1f, 2f, 3f, 4f)))),
            actual = actual,
        )
    }

    @Test
    fun `train_axis2=中間の入力を含めてdeltaを列方向に分配する`() = networkScopeTestRule {
        val target = ConcatD3(outputI = 1, outputJ = 1, outputK = 4, axis = 2)
        val actual = with(target) {
            _train(inputs = listOf(a2, b2, c2) as List<Batch<IOType>>, env = GraphEnv(), calcDelta = { it })
        }

        assertContentEquals(expected = a2, actual = actual[0] as Batch<IOType.D3>)
        assertContentEquals(expected = b2, actual = actual[1] as Batch<IOType.D3>)
        assertContentEquals(expected = c2, actual = actual[2] as Batch<IOType.D3>)
    }
}
