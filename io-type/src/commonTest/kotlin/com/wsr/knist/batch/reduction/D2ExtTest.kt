@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.reduction
import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class D2ExtTest {
    @Test
    fun `sum=D2バッチの各要素の合計`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.sum()
        assertContentEquals(IOType.d0(6f), result[0])
        assertContentEquals(IOType.d0(22f), result[1])
    }

    @Test
    fun `sum_axis0=D2バッチのaxis0合計`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.sum(axis = 0)
        assertContentEquals(IOType.d1(listOf(2f, 4f)), result[0])
        assertContentEquals(IOType.d1(listOf(10f, 12f)), result[1])
    }

    @Test
    fun `sum_axis1=D2バッチのaxis1合計`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.sum(axis = 1)
        assertContentEquals(IOType.d1(listOf(1f, 5f)), result[0])
        assertContentEquals(IOType.d1(listOf(9f, 13f)), result[1])
    }

    @Test
    fun `max=D2バッチの各要素の最大値`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.max()
        assertContentEquals(IOType.d0(3f), result[0])
        assertContentEquals(IOType.d0(7f), result[1])
    }

    @Test
    fun `min=D2バッチの各要素の最小値`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(2, 2) { i, j -> i * 2f + j },
            IOType.d2(2, 2) { i, j -> i * 2f + j + 4f },
        )
        val result = batch.min()
        assertContentEquals(IOType.d0(0f), result[0])
        assertContentEquals(IOType.d0(4f), result[1])
    }

    @Test
    fun `maxIndex_axis0=D2バッチのaxis0最大値インデックス`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(shape = listOf(2, 3), value = listOf(3f, 1f, 4f, 1f, 5f, 2f)),
            IOType.d2(shape = listOf(2, 3), value = listOf(2f, 7f, 1f, 6f, 0f, 3f)),
        )
        val result = batch.maxIndex(axis = 0)
        assertContentEquals(IOType.d1(listOf(0f, 1f, 0f)), result[0])
        assertContentEquals(IOType.d1(listOf(1f, 0f, 1f)), result[1])
    }

    @Test
    fun `maxIndex_axis1=D2バッチのaxis1最大値インデックス`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(shape = listOf(2, 3), value = listOf(3f, 1f, 4f, 1f, 5f, 2f)),
            IOType.d2(shape = listOf(2, 3), value = listOf(2f, 7f, 1f, 6f, 0f, 3f)),
        )
        val result = batch.maxIndex(axis = 1)
        assertContentEquals(IOType.d1(listOf(2f, 1f)), result[0])
        assertContentEquals(IOType.d1(listOf(1f, 0f)), result[1])
    }

    @Test
    fun `topK_k=1_axis0=D2バッチのaxis0最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(shape = listOf(2, 3), value = listOf(3f, 1f, 4f, 1f, 5f, 2f)),
            IOType.d2(shape = listOf(2, 3), value = listOf(2f, 7f, 1f, 6f, 0f, 3f)),
        )
        val result = batch.topK(k = 1, axis = 0)
        assertContentEquals(IOType.d1(listOf(0f, 1f, 0f)), result[0])
        assertContentEquals(IOType.d1(listOf(1f, 0f, 1f)), result[1])
    }

    @Test
    fun `topK_k=1_axis1=D2バッチのaxis1最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(shape = listOf(2, 3), value = listOf(3f, 1f, 4f, 1f, 5f, 2f)),
            IOType.d2(shape = listOf(2, 3), value = listOf(2f, 7f, 1f, 6f, 0f, 3f)),
        )
        val result = batch.topK(k = 1, axis = 1)
        assertContentEquals(IOType.d1(listOf(2f, 1f)), result[0])
        assertContentEquals(IOType.d1(listOf(1f, 0f)), result[1])
    }

    @Test
    fun `topP_p=0_axis0=D2バッチのaxis0最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(shape = listOf(2, 3), value = listOf(3f, 1f, 4f, 1f, 5f, 2f)),
            IOType.d2(shape = listOf(2, 3), value = listOf(2f, 7f, 1f, 6f, 0f, 3f)),
        )
        val result = batch.topP(p = 0f, axis = 0)
        assertContentEquals(IOType.d1(listOf(0f, 1f, 0f)), result[0])
        assertContentEquals(IOType.d1(listOf(1f, 0f, 1f)), result[1])
    }

    @Test
    fun `topP_p=0_axis1=D2バッチのaxis1最大値インデックスを選択`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d2(shape = listOf(2, 3), value = listOf(3f, 1f, 4f, 1f, 5f, 2f)),
            IOType.d2(shape = listOf(2, 3), value = listOf(2f, 7f, 1f, 6f, 0f, 3f)),
        )
        val result = batch.topP(p = 0f, axis = 1)
        assertContentEquals(IOType.d1(listOf(2f, 1f)), result[0])
        assertContentEquals(IOType.d1(listOf(1f, 0f)), result[1])
    }
}
