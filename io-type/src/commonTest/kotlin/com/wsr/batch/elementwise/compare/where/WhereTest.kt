@file:Suppress("NonAsciiCharacters")

package com.wsr.batch.elementwise.compare.where

import com.wsr.assertContentEquals
import com.wsr.batch.batchOf
import com.wsr.batch.elementwise.compare.gt
import com.wsr.batch.get
import com.wsr.core.IOType
import com.wsr.core.d0
import com.wsr.core.d1
import com.wsr.ioTypeTestRule
import kotlin.test.Test

class WhereTest {
    @Test
    fun `where_D0_FloatToFloat=条件に基づきD0バッチを選択`() = ioTypeTestRule {
        val condition = batchOf(IOType.d0(1f), IOType.d0(0f), IOType.d0(1f))
        val result = where(condition, onTrue = 5f, onFalse = 0f)
        assertContentEquals(IOType.d0(5f), result[0])
        assertContentEquals(IOType.d0(0f), result[1])
        assertContentEquals(IOType.d0(5f), result[2])
    }

    @Test
    fun `where_D0_instance=D0バッチのインスタンスメソッドwhere`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val condition = batchOf(IOType.d0(1f), IOType.d0(0f), IOType.d0(1f))
        val result = batch.where(condition, onTrue = 99f, onFalse = 0f)
        assertContentEquals(IOType.d0(99f), result[0])
        assertContentEquals(IOType.d0(0f), result[1])
        assertContentEquals(IOType.d0(99f), result[2])
    }

    @Test
    fun `where_D1_FloatToFloat=条件に基づきD1バッチを選択`() = ioTypeTestRule {
        val condition = batchOf(
            IOType.d1(listOf(1f, 0f, 1f)),
            IOType.d1(listOf(0f, 1f, 0f)),
        )
        val result = where(condition, onTrue = 5f, onFalse = 0f)
        assertContentEquals(IOType.d1(listOf(5f, 0f, 5f)), result[0])
        assertContentEquals(IOType.d1(listOf(0f, 5f, 0f)), result[1])
    }

    @Test
    fun `where_D0_withLambda=ラムダ条件でD0バッチを選択`() = ioTypeTestRule {
        val batch = batchOf(IOType.d0(1f), IOType.d0(2f), IOType.d0(3f))
        val result = batch.where(onTrue = 99f, onFalse = 0f) { it gt 1f }
        assertContentEquals(IOType.d0(0f), result[0])
        assertContentEquals(IOType.d0(99f), result[1])
        assertContentEquals(IOType.d0(99f), result[2])
    }
}
