package com.wsr.base

interface IDataBufferGenerator {
    fun create(size: Int): DataBuffer
    fun create(value: FloatArray): DataBuffer
}

object DataBufferGenerator : IDataBufferGenerator {
    private var instance = Default.generator

    fun set(generator: IDataBufferGenerator) {
        instance = generator
    }

    override fun create(size: Int): DataBuffer = instance.create(size)
    override fun create(value: FloatArray): DataBuffer = instance.create(value)
}
