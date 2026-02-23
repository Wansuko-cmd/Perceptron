package com.wsr.gpu

import com.wsr.base.IBackend
import com.wsr.base.KotlinBackend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.IDataBufferGenerator
import com.wsr.base.loadNativeLibrary
import java.lang.ref.Cleaner

private const val LIB_PATH = "gpu"
private const val LIB_NAME = "gpu"

actual fun loadGPUBackend(): IBackend? {
    val isSuccess = loadNativeLibrary(path = LIB_PATH, name = LIB_NAME)
    return if (isSuccess) GPUBackend() else null
}

class GPUBackend : IBackend by KotlinBackend {
    private val context = JContext().allocate()
    private val buffer = JBuffer()
    override val generator: IDataBufferGenerator = GPUJvmBuffer.createGenerator(context, buffer)

    private val math = JMath()
    private val transpose = JTranspose()

    init {
        val ptr = context
        cleaner.register(this) { JContext().release(ptr) }
    }

    override fun exp(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.exp(x.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    override fun ln(x: DataBuffer, e: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.ln(x.toGPUBuffer().ptr, e, result.ptr, context)
        return result
    }

    override fun pow(x: DataBuffer, n: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.pow(x.toGPUBuffer().ptr, n, result.ptr, context)
        return result
    }

    override fun sqrt(x: DataBuffer, e: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.sqrt(x.toGPUBuffer().ptr, e, result.ptr, context)
        return result
    }

    override fun transpose(x: DataBuffer, xi: Int, xj: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        transpose.transposeD2(x.toGPUBuffer().ptr, xi, xj, result.ptr, context)
        return result
    }

    override fun transpose(x: DataBuffer, xi: Int, xj: Int, xk: Int, axisI: Int, axisJ: Int, axisK: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        transpose.transposeD3(x.toGPUBuffer().ptr, xi, xj, xk, axisI, axisJ, axisK, result.ptr, context)
        return result
    }

    override fun transpose(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        axisI: Int,
        axisJ: Int,
        axisK: Int,
        axisL: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        transpose.transposeD4(x.toGPUBuffer().ptr, xi, xj, xk, xl, axisI, axisJ, axisK, axisL, result.ptr, context)
        return result
    }

    private fun GPUJvmBuffer.Companion.create(size: Int) = GPUJvmBuffer.create(size, context, buffer)

    private fun DataBuffer.toGPUBuffer(): GPUJvmBuffer = toGPUBuffer(context, buffer)

    companion object {
        private val cleaner = Cleaner.create()
    }
}
