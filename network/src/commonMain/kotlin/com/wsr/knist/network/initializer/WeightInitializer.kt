package com.wsr.knist.network.initializer

import com.wsr.knist.core.IOType

/**
 * @param input 層の入力の形状
 * @param output 層の出力の形状
 * @param [size | x,y,z | i,j,k,z] 重みの数
 */
interface WeightInitializer {
    fun d1(input: List<Int>, output: List<Int>, size: Int): IOType.D1.Global
    fun d2(input: List<Int>, output: List<Int>, i: Int, j: Int): IOType.D2.Global
    fun d3(input: List<Int>, output: List<Int>, i: Int, j: Int, k: Int): IOType.D3.Global
    fun d4(input: List<Int>, output: List<Int>, i: Int, j: Int, k: Int, l: Int): IOType.D4.Global
}
