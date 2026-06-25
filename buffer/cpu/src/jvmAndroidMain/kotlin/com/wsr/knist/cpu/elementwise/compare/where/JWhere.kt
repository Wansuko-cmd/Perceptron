package com.wsr.knist.cpu.elementwise.compare.where

import java.nio.ByteBuffer

object JWhere {
    external fun whereD0ToD0(condition: ByteBuffer, x: Float, y: Float, result: ByteBuffer)
    external fun whereD0ToD1(condition: ByteBuffer, x: Float, y: ByteBuffer, result: ByteBuffer)
    external fun whereD1ToD0(condition: ByteBuffer, x: ByteBuffer, y: Float, result: ByteBuffer)
    external fun whereD1ToD1(condition: ByteBuffer, x: ByteBuffer, y: ByteBuffer, result: ByteBuffer)
}
