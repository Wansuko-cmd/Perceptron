@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.batch.elementwise.operation
import com.wsr.knist.assertContentEquals
import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.elementwise.operation.plus.plus
import com.wsr.knist.batch.get
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d0
import com.wsr.knist.core.d1
import com.wsr.knist.ioTypeTestRule
import kotlin.test.Test

class PlusTest {
    @Test
    fun `D0バッチ+スカラー`() = ioTypeTestRule {
        val batch = Batch.of(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val result = batch + 10f
        assertContentEquals(IOType.d0(11f), result[0])
        assertContentEquals(IOType.d0(12f), result[1])
        assertContentEquals(IOType.d0(13f), result[2])
    }

    @Test
    fun `スカラー+D0バッチ`() = ioTypeTestRule {
        val batch = Batch.of(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val result = 10f + batch
        assertContentEquals(IOType.d0(11f), result[0])
        assertContentEquals(IOType.d0(12f), result[1])
        assertContentEquals(IOType.d0(13f), result[2])
    }

    @Test
    fun `D0バッチ+D0バッチ`() = ioTypeTestRule {
        val batch1 = Batch.of(IOType.d0(1f), IOType.d0(2f))
        val batch2 = Batch.of(IOType.d0(10f), IOType.d0(20f))
        val result = batch1 + batch2
        assertContentEquals(IOType.d0(11f), result[0])
        assertContentEquals(IOType.d0(22f), result[1])
    }

    @Test
    fun `D0バッチ+D1バッチ=ブロードキャスト`() = ioTypeTestRule {
        val batchD0 = Batch.of(IOType.d0(10f), IOType.d0(20f))
        val batchD1 = Batch.of(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = batchD0 + batchD1
        assertContentEquals(IOType.d1(listOf(11f, 12f, 13f)), result[0])
        assertContentEquals(IOType.d1(listOf(24f, 25f, 26f)), result[1])
    }

    @Test
    fun `D1バッチ+スカラー`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = batch + 10f
        assertContentEquals(IOType.d1(listOf(11f, 12f, 13f)), result[0])
        assertContentEquals(IOType.d1(listOf(14f, 15f, 16f)), result[1])
    }

    @Test
    fun `スカラー+D1バッチ`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val result = 10f + batch
        assertContentEquals(IOType.d1(listOf(11f, 12f, 13f)), result[0])
        assertContentEquals(IOType.d1(listOf(14f, 15f, 16f)), result[1])
    }

    @Test
    fun `D1バッチ+D1バッチ`() = ioTypeTestRule {
        val batch1 = Batch.of(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val batch2 = Batch.of(
            IOType.d1(listOf(10f, 20f, 30f)),
            IOType.d1(listOf(40f, 50f, 60f)),
        )
        val result = batch1 + batch2
        assertContentEquals(IOType.d1(listOf(11f, 22f, 33f)), result[0])
        assertContentEquals(IOType.d1(listOf(44f, 55f, 66f)), result[1])
    }

    @Test
    fun `D1バッチ+D1=ブロードキャスト`() = ioTypeTestRule {
        val batch = Batch.of(
            IOType.d1(listOf(1f, 2f, 3f)),
            IOType.d1(listOf(4f, 5f, 6f)),
        )
        val d1 = IOType.d1(listOf(10f, 10f, 10f))
        val result = batch + d1
        assertContentEquals(IOType.d1(listOf(11f, 12f, 13f)), result[0])
        assertContentEquals(IOType.d1(listOf(14f, 15f, 16f)), result[1])
    }
}
