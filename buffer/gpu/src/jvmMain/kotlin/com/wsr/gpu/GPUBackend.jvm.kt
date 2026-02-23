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

    init {
        val ptr = context
        cleaner.register(this) { JContext().release(ptr) }
    }

    override fun exp(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size, context, buffer)
        math.exp(x.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    override fun ln(x: DataBuffer, e: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size, context, buffer)
        math.ln(x.toGPUBuffer().ptr, e, result.ptr, context)
        return result
    }

    override fun pow(x: DataBuffer, n: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size, context, buffer)
        math.pow(x.toGPUBuffer().ptr, n, result.ptr, context)
        return result
    }

    override fun sqrt(x: DataBuffer, e: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size, context, buffer)
        math.sqrt(x.toGPUBuffer().ptr, e, result.ptr, context)
        return result
    }

    private fun DataBuffer.toGPUBuffer(): GPUJvmBuffer = toGPUBuffer(context, buffer)

    companion object {
        private val cleaner = Cleaner.create()
    }
}
