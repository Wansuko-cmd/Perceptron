package com.wsr.base.data

import kotlin.jvm.JvmName

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

fun DataBuffer.Companion.create(size: Int) = DataBufferGenerator.create(size)

fun DataBuffer.Companion.create(value: FloatArray) = DataBufferGenerator.create(value)

@JvmName("createWithElements")
fun DataBuffer.Companion.create(vararg elements: Float) = DataBufferGenerator.create(elements)
