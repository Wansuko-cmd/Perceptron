@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.reduction
import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class D4ExtTest {

    private val d4v0 = IOType.d4(
        shape = listOf(2, 2, 2, 2),
        value = listOf(
            1f, 2f,
            3f, 4f,
            5f, 6f,
            7f, 8f,
            9f, 10f,
            11f, 12f,
            13f, 14f,
            15f, 16f,
        ),
    )

    private val d4v1 = IOType.d4(
        shape = listOf(2, 2, 2, 2),
        value = listOf(
            16f, 15f,
            14f, 13f,
            12f, 11f,
            10f, 9f,
            8f, 7f,
            6f, 5f,
            4f, 3f,
            2f, 1f,
        ),
    )

    @Test
    fun `max_axis0=D4バッチのaxis0最大値`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.max(axis = 0)
        assertContentEquals(
            IOType.d3(shape = listOf(2, 2, 2), value = listOf(9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f)),
            result[0],
        )
        assertContentEquals(
            IOType.d3(shape = listOf(2, 2, 2), value = listOf(16f, 15f, 14f, 13f, 12f, 11f, 10f, 9f)),
            result[1],
        )
    }

    @Test
    fun `max_axis1=D4バッチのaxis1最大値`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.max(axis = 1)
        assertContentEquals(
            IOType.d3(shape = listOf(2, 2, 2), value = listOf(5f, 6f, 7f, 8f, 13f, 14f, 15f, 16f)),
            result[0],
        )
        assertContentEquals(
            IOType.d3(shape = listOf(2, 2, 2), value = listOf(16f, 15f, 14f, 13f, 8f, 7f, 6f, 5f)),
            result[1],
        )
    }

    @Test
    fun `max_axis2=D4バッチのaxis2最大値`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.max(axis = 2)
        assertContentEquals(
            IOType.d3(shape = listOf(2, 2, 2), value = listOf(3f, 4f, 7f, 8f, 11f, 12f, 15f, 16f)),
            result[0],
        )
        assertContentEquals(
            IOType.d3(shape = listOf(2, 2, 2), value = listOf(16f, 15f, 12f, 11f, 8f, 7f, 4f, 3f)),
            result[1],
        )
    }

    @Test
    fun `max_axis3=D4バッチのaxis3最大値`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.max(axis = 3)
        assertContentEquals(
            IOType.d3(shape = listOf(2, 2, 2), value = listOf(2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f)),
            result[0],
        )
        assertContentEquals(
            IOType.d3(shape = listOf(2, 2, 2), value = listOf(16f, 14f, 12f, 10f, 8f, 6f, 4f, 2f)),
            result[1],
        )
    }

    // d4v0: 1..16 (i outer, l inner) → max index は常に 1
    // d4v1: 16..1 (逆順)              → max index は常に 0
    private fun d4AllOnes() = IOType.d3(shape = listOf(2, 2, 2), value = List(8) { 1f })
    private fun d4AllZeros() = IOType.d3(shape = listOf(2, 2, 2), value = List(8) { 0f })

    @Test
    fun `maxIndex_axis0=D4バッチのaxis0最大値インデックス`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.maxIndex(axis = 0)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `maxIndex_axis1=D4バッチのaxis1最大値インデックス`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.maxIndex(axis = 1)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `maxIndex_axis2=D4バッチのaxis2最大値インデックス`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.maxIndex(axis = 2)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `maxIndex_axis3=D4バッチのaxis3最大値インデックス`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.maxIndex(axis = 3)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `topK_k=1_axis0=D4バッチのaxis0最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.topK(k = 1, axis = 0)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `topK_k=1_axis1=D4バッチのaxis1最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.topK(k = 1, axis = 1)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `topK_k=1_axis2=D4バッチのaxis2最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.topK(k = 1, axis = 2)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `topK_k=1_axis3=D4バッチのaxis3最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.topK(k = 1, axis = 3)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `topP_p=0_axis0=D4バッチのaxis0最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.topP(p = 0f, axis = 0)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `topP_p=0_axis1=D4バッチのaxis1最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.topP(p = 0f, axis = 1)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `topP_p=0_axis2=D4バッチのaxis2最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.topP(p = 0f, axis = 2)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }

    @Test
    fun `topP_p=0_axis3=D4バッチのaxis3最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(d4v0, d4v1)
        val result = batch.topP(p = 0f, axis = 3)
        assertContentEquals(d4AllOnes(), result[0])
        assertContentEquals(d4AllZeros(), result[1])
    }
}
