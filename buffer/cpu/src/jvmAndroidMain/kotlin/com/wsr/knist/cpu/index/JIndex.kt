package com.wsr.knist.cpu.index


object JIndex {
    external fun gather(x: Long, y: Long, i: Int, j: Int, k: Int, result: Long)
    external fun scatterAdd(x: Long, y: Long, i: Int, j: Int, k: Int, b: Int, result: Long)
}
