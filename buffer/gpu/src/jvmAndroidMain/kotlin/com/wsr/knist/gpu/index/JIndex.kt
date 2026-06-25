package com.wsr.knist.gpu.index

object JIndex {
    external fun gather(x: Long, y: Long, i: Int, j: Int, k: Int, result: Long, runtime: Long)
    external fun scatterAdd(x: Long, y: Long, i: Int, j: Int, k: Int, b: Int, result: Long, runtime: Long)
}
