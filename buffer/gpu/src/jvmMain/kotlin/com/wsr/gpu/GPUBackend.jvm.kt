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
    private val matMul = JMatMul()
    private val collection = JCollection()
    private val operation = JOperation()
    private val transpose = JTranspose()

    init {
        val ptr = context
        cleaner.register(this) { JContext().release(ptr) }
    }

    // 0次元
    override fun plus(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.plusD0ToD1(x, y.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    // 1次元
    override fun plus(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.plusD1ToD0(x.toGPUBuffer().ptr, y, result.ptr, context)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.plusD1ToD1(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.plusD1ToD2(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, yi, yj, axis, result.ptr, context)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.plusD1ToD3(
            x.toGPUBuffer().ptr,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    // 2次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.plusD2ToD1(x.toGPUBuffer().ptr, xi, xj, y.toGPUBuffer().ptr, axis, result.ptr, context)
        return result
    }

    override fun plus(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.plusD2ToD3(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    // 3次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.plusD3ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    override fun plus(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.plusD3ToD2(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    override fun plus(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        yl: Int,
        axis1: Int,
        axis2: Int,
        axis3: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.plusD3ToD4(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            yl,
            axis1,
            axis2,
            axis3,
            result.ptr,
            context,
        )
        return result
    }

    // 4次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.plusD4ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    override fun plus(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.plusD4ToD2(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    override fun plus(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        axis1: Int,
        axis2: Int,
        axis3: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.plusD4ToD3(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            axis3,
            result.ptr,
            context,
        )
        return result
    }

    // 0次元
    override fun minus(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.minusD0ToD1(x, y.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    // 1次元
    override fun minus(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.minusD1ToD0(x.toGPUBuffer().ptr, y, result.ptr, context)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.minusD1ToD1(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.minusD1ToD2(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, yi, yj, axis, result.ptr, context)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.minusD1ToD3(
            x.toGPUBuffer().ptr,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    // 2次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.minusD2ToD1(x.toGPUBuffer().ptr, xi, xj, y.toGPUBuffer().ptr, axis, result.ptr, context)
        return result
    }

    override fun minus(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.minusD2ToD3(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    // 3次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.minusD3ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    override fun minus(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.minusD3ToD2(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    override fun minus(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        yl: Int,
        axis1: Int,
        axis2: Int,
        axis3: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.minusD3ToD4(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            yl,
            axis1,
            axis2,
            axis3,
            result.ptr,
            context,
        )
        return result
    }

    // 4次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.minusD4ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    override fun minus(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.minusD4ToD2(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    override fun minus(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        axis1: Int,
        axis2: Int,
        axis3: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.minusD4ToD3(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            axis3,
            result.ptr,
            context,
        )
        return result
    }

    // 0次元
    override fun times(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.timesD0ToD1(x, y.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    // 1次元
    override fun times(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.timesD1ToD0(x.toGPUBuffer().ptr, y, result.ptr, context)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.timesD1ToD1(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.timesD1ToD2(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, yi, yj, axis, result.ptr, context)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.timesD1ToD3(
            x.toGPUBuffer().ptr,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    // 2次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.timesD2ToD1(x.toGPUBuffer().ptr, xi, xj, y.toGPUBuffer().ptr, axis, result.ptr, context)
        return result
    }

    override fun times(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.timesD2ToD3(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    // 3次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.timesD3ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    override fun times(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.timesD3ToD2(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    override fun times(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        yl: Int,
        axis1: Int,
        axis2: Int,
        axis3: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.timesD3ToD4(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            yl,
            axis1,
            axis2,
            axis3,
            result.ptr,
            context,
        )
        return result
    }

    // 4次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.timesD4ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    override fun times(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.timesD4ToD2(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    override fun times(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        axis1: Int,
        axis2: Int,
        axis3: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.timesD4ToD3(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            axis3,
            result.ptr,
            context,
        )
        return result
    }

    // 0次元
    override fun div(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.divD0ToD1(x, y.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    // 1次元
    override fun div(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.divD1ToD0(x.toGPUBuffer().ptr, y, result.ptr, context)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.divD1ToD1(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.divD1ToD2(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, yi, yj, axis, result.ptr, context)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.divD1ToD3(
            x.toGPUBuffer().ptr,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    // 2次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.divD2ToD1(x.toGPUBuffer().ptr, xi, xj, y.toGPUBuffer().ptr, axis, result.ptr, context)
        return result
    }

    override fun div(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.divD2ToD3(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    // 3次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.divD3ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    override fun div(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.divD3ToD2(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    override fun div(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        yl: Int,
        axis1: Int,
        axis2: Int,
        axis3: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        operation.divD3ToD4(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            yl,
            axis1,
            axis2,
            axis3,
            result.ptr,
            context,
        )
        return result
    }

    // 4次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.divD4ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            context,
        )
        return result
    }

    override fun div(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        axis1: Int,
        axis2: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.divD4ToD2(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            axis1,
            axis2,
            result.ptr,
            context,
        )
        return result
    }

    override fun div(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        y: DataBuffer,
        yi: Int,
        yj: Int,
        yk: Int,
        axis1: Int,
        axis2: Int,
        axis3: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        operation.divD4ToD3(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            axis3,
            result.ptr,
            context,
        )
        return result
    }

    override fun inner(x: DataBuffer, y: DataBuffer, b: Int): DataBuffer {
        val result = GPUJvmBuffer.create(b)
        matMul.matMul(
            x.toGPUBuffer().ptr, false,
            y.toGPUBuffer().ptr, false,
            1, 1, x.size / b, b,
            result.ptr,
            context,
        )
        return result
    }

    override fun matMul(x: DataBuffer, y: DataBuffer, transY: Boolean, n: Int, k: Int): DataBuffer {
        val result = GPUJvmBuffer.create(n)
        matMul.matMul(
            x.toGPUBuffer().ptr, false,
            y.toGPUBuffer().ptr, transY,
            1, n, k, 1,
            result.ptr,
            context,
        )
        return result
    }

    override fun matMul(x: DataBuffer, transX: Boolean, y: DataBuffer, m: Int, k: Int): DataBuffer {
        val result = GPUJvmBuffer.create(m)
        matMul.matMul(
            x.toGPUBuffer().ptr, transX,
            y.toGPUBuffer().ptr, false,
            m, 1, k, 1,
            result.ptr,
            context,
        )
        return result
    }

    override fun matMul(
        x: DataBuffer,
        transX: Boolean,
        y: DataBuffer,
        transY: Boolean,
        m: Int,
        n: Int,
        k: Int,
        b: Int,
    ): DataBuffer {
        val result = GPUJvmBuffer.create(b * m * n)
        matMul.matMul(
            x.toGPUBuffer().ptr, transX,
            y.toGPUBuffer().ptr, transY,
            m, n, k, b,
            result.ptr,
            context,
        )
        return result
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

    override fun average(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        collection.averageD1(x.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    override fun average(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.averageD2(x.toGPUBuffer().ptr, xi, xj, axis, result.ptr, context)
        return result
    }

    override fun average(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        collection.averageD3(x.toGPUBuffer().ptr, xi, xj, xk, axis, result.ptr, context)
        return result
    }

    override fun max(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        collection.maxD1(x.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    override fun max(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.maxD2(x.toGPUBuffer().ptr, xi, xj, axis, result.ptr, context)
        return result
    }

    override fun max(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        collection.maxD3(x.toGPUBuffer().ptr, xi, xj, xk, axis, result.ptr, context)
        return result
    }

    override fun min(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        collection.minD1(x.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    override fun min(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.minD2(x.toGPUBuffer().ptr, xi, xj, axis, result.ptr, context)
        return result
    }

    override fun min(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        collection.minD3(x.toGPUBuffer().ptr, xi, xj, xk, axis, result.ptr, context)
        return result
    }

    override fun sum(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        collection.sumD1(x.toGPUBuffer().ptr, result.ptr, context)
        return result
    }

    override fun sum(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.sumD2(x.toGPUBuffer().ptr, xi, xj, axis, result.ptr, context)
        return result
    }

    override fun sum(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        collection.sumD3(x.toGPUBuffer().ptr, xi, xj, xk, axis, result.ptr, context)
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
