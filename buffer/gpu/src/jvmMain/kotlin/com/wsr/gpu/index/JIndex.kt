package com.wsr.gpu.index

class JIndex {
    external fun gather(x: Long, y: Long, i: Int, j: Int, k: Int, result: Long, runtime: Long)
    external fun scatterAdd(x: Long, y: Long, i: Int, j: Int, k: Int, b: Int, result: Long, runtime: Long)
}
