package com.wsr.knist.gpu.reduction

class JReduction {
    external fun averageD1(x: Long, result: Long, runtime: Long)
    external fun averageD2(x: Long, xi: Int, xj: Int, axis: Int, result: Long, runtime: Long)
    external fun averageD3(x: Long, xi: Int, xj: Int, xk: Int, axis: Int, result: Long, runtime: Long)

    external fun maxD1(x: Long, result: Long, runtime: Long)
    external fun maxD2(x: Long, xi: Int, xj: Int, axis: Int, result: Long, runtime: Long)
    external fun maxD3(x: Long, xi: Int, xj: Int, xk: Int, axis: Int, result: Long, runtime: Long)

    external fun minD1(x: Long, result: Long, runtime: Long)
    external fun minD2(x: Long, xi: Int, xj: Int, axis: Int, result: Long, runtime: Long)
    external fun minD3(x: Long, xi: Int, xj: Int, xk: Int, axis: Int, result: Long, runtime: Long)

    external fun sumD1(x: Long, result: Long, runtime: Long)
    external fun sumD2(x: Long, xi: Int, xj: Int, axis: Int, result: Long, runtime: Long)
    external fun sumD3(x: Long, xi: Int, xj: Int, xk: Int, axis: Int, result: Long, runtime: Long)
}
