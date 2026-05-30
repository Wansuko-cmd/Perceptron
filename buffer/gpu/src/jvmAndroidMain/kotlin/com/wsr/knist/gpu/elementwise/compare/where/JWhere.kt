package com.wsr.knist.gpu.elementwise.compare.where

class JWhere {
    external fun whereD0ToD0(condition: Long, x: Float, y: Float, result: Long, runtime: Long)
    external fun whereD0ToD1(condition: Long, x: Float, y: Long, result: Long, runtime: Long)
    external fun whereD1ToD0(condition: Long, x: Long, y: Float, result: Long, runtime: Long)
    external fun whereD1ToD1(condition: Long, x: Long, y: Long, result: Long, runtime: Long)
}
