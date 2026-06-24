package com.wsr.knist.gpu

import com.wsr.knist.base.IBackend
import com.wsr.knist.base.KotlinBackend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.base.data.IDataBufferGenerator
import com.wsr.knist.base.data.size
import com.wsr.knist.gpu.elementwise.compare.JCompare
import com.wsr.knist.gpu.elementwise.compare.where.JWhere
import com.wsr.knist.gpu.elementwise.math.JMath
import com.wsr.knist.gpu.elementwise.operation.JDiv
import com.wsr.knist.gpu.elementwise.operation.JMinus
import com.wsr.knist.gpu.elementwise.operation.JPlus
import com.wsr.knist.gpu.elementwise.operation.JTimes
import com.wsr.knist.gpu.index.JIndex
import com.wsr.knist.gpu.linalg.JMatMul
import com.wsr.knist.gpu.reduction.JReduction
import com.wsr.knist.gpu.shape.JShape
import java.lang.ref.Cleaner

class GPUBackend : IBackend by KotlinBackend {
    private val jRuntime = JRuntime()
    private val runtime = jRuntime.allocate()
    private val buffer = JBuffer()
    override val generator: IDataBufferGenerator = GPUJvmBuffer.createGenerator(runtime = runtime, buffer)

    private val math = JMath()
    private val matMul = JMatMul()
    private val reduction = JReduction()
    private val index = JIndex()
    private val plus = JPlus()
    private val minus = JMinus()
    private val times = JTimes()
    private val div = JDiv()
    private val shape = JShape()
    private val compare = JCompare()
    private val jWhere = JWhere()

    init {
        val ptr = runtime
        cleaner.register(this) { JRuntime().release(ptr) }
    }

    // 0次元
    override fun plus(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        plus.plusD0ToD1(x = x, y = y.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    // 1次元
    override fun plus(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        plus.plusD1ToD0(x = x.toGPUBuffer().ptr, y = y, result = result.ptr, runtime = runtime)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        plus.plusD1ToD1(x = x.toGPUBuffer().ptr, y = y.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        plus.plusD1ToD2(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        plus.plusD1ToD3(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 2次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        plus.plusD2ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
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
        plus.plusD2ToD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 3次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        plus.plusD3ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
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
        plus.plusD3ToD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
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
        plus.plusD3ToD4(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            yl = yl,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 4次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        plus.plusD4ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
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
        plus.plusD4ToD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
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
        plus.plusD4ToD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 0次元
    override fun minus(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        minus.minusD0ToD1(x = x, y = y.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    // 1次元
    override fun minus(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        minus.minusD1ToD0(x = x.toGPUBuffer().ptr, y = y, result = result.ptr, runtime = runtime)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        minus.minusD1ToD1(x = x.toGPUBuffer().ptr, y = y.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        minus.minusD1ToD2(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        minus.minusD1ToD3(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 2次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        minus.minusD2ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
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
        minus.minusD2ToD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 3次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        minus.minusD3ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
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
        minus.minusD3ToD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
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
        minus.minusD3ToD4(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            yl = yl,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 4次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        minus.minusD4ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
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
        minus.minusD4ToD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
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
        minus.minusD4ToD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 0次元
    override fun times(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        times.timesD0ToD1(x = x, y = y.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    // 1次元
    override fun times(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        times.timesD1ToD0(x = x.toGPUBuffer().ptr, y = y, result = result.ptr, runtime = runtime)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        times.timesD1ToD1(x = x.toGPUBuffer().ptr, y = y.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        times.timesD1ToD2(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        times.timesD1ToD3(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 2次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        times.timesD2ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
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
        times.timesD2ToD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 3次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        times.timesD3ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
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
        times.timesD3ToD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
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
        times.timesD3ToD4(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            yl = yl,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 4次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        times.timesD4ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
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
        times.timesD4ToD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
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
        times.timesD4ToD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 0次元
    override fun div(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        div.divD0ToD1(x = x, y = y.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    // 1次元
    override fun div(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        div.divD1ToD0(x = x.toGPUBuffer().ptr, y = y, result = result.ptr, runtime = runtime)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        div.divD1ToD1(x = x.toGPUBuffer().ptr, y = y.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        div.divD1ToD2(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        div.divD1ToD3(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 2次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        div.divD2ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
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
        div.divD2ToD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 3次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        div.divD3ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
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
        div.divD3ToD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
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
        div.divD3ToD4(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            yl = yl,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    // 4次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        div.divD4ToD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
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
        div.divD4ToD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.ptr,
            runtime = runtime,
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
        div.divD4ToD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toGPUBuffer().ptr,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun inner(x: DataBuffer, y: DataBuffer, b: Int): DataBuffer {
        val result = GPUJvmBuffer.create(b)
        matMul.matMul(
            x = x.toGPUBuffer().ptr,
            transX = false,
            y = y.toGPUBuffer().ptr,
            transY = false,
            m = 1,
            n = 1,
            k = x.size / b,
            b = b,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun matMul(x: DataBuffer, y: DataBuffer, transY: Boolean, n: Int, k: Int): DataBuffer {
        val result = GPUJvmBuffer.create(n)
        matMul.matMul(
            x = x.toGPUBuffer().ptr,
            transX = false,
            y = y.toGPUBuffer().ptr,
            transY = transY,
            m = 1,
            n = n,
            k = k,
            b = 1,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun matMul(x: DataBuffer, transX: Boolean, y: DataBuffer, m: Int, k: Int): DataBuffer {
        val result = GPUJvmBuffer.create(m)
        matMul.matMul(
            x = x.toGPUBuffer().ptr,
            transX = transX,
            y = y.toGPUBuffer().ptr,
            transY = false,
            m = m,
            n = 1,
            k = k,
            b = 1,
            result = result.ptr,
            runtime = runtime,
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
            x = x.toGPUBuffer().ptr,
            transX = transX,
            y = y.toGPUBuffer().ptr,
            transY = transY,
            m = m,
            n = n,
            k = k,
            b = b,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun exp(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.exp(x = x.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun ln(x: DataBuffer, e: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.ln(x = x.toGPUBuffer().ptr, e = e, result = result.ptr, runtime = runtime)
        return result
    }

    override fun sigmoid(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.sigmoid(x = x.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun pow(x: DataBuffer, n: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.pow(x = x.toGPUBuffer().ptr, n = n, result = result.ptr, runtime = runtime)
        return result
    }

    override fun sqrt(x: DataBuffer, e: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.sqrt(x = x.toGPUBuffer().ptr, e = e, result = result.ptr, runtime = runtime)
        return result
    }

    override fun average(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        reduction.averageD1(x = x.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun average(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        reduction.averageD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
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
        reduction.averageD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun max(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        reduction.maxD1(x = x.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun max(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        reduction.maxD2(x = x.toGPUBuffer().ptr, xi = xi, xj = xj, axis = axis, result = result.ptr, runtime = runtime)
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
        reduction.maxD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun min(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        reduction.minD1(x = x.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun min(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        reduction.minD2(x = x.toGPUBuffer().ptr, xi = xi, xj = xj, axis = axis, result = result.ptr, runtime = runtime)
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
        reduction.minD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun sum(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        reduction.sumD1(x = x.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun sum(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        reduction.sumD2(x = x.toGPUBuffer().ptr, xi = xi, xj = xj, axis = axis, result = result.ptr, runtime = runtime)
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
        reduction.sumD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun transpose(x: DataBuffer, xi: Int, xj: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        shape.transposeD2(x = x.toGPUBuffer().ptr, xi = xi, xj = xj, result = result.ptr, runtime = runtime)
        return result
    }

    override fun transpose(x: DataBuffer, xi: Int, xj: Int, xk: Int, axisI: Int, axisJ: Int, axisK: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        shape.transposeD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            axisI = axisI,
            axisJ = axisJ,
            axisK = axisK,
            result = result.ptr,
            runtime = runtime,
        )
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
        shape.transposeD4(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            axisI = axisI,
            axisJ = axisJ,
            axisK = axisK,
            axisL = axisL,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun slice(x: DataBuffer, indices: IntProgression): DataBuffer {
        val result = GPUJvmBuffer.create(kotlin.math.min(x.size, indices.size))
        shape.sliceD1(
            x = x.toGPUBuffer().ptr,
            start = indices.first,
            end = indices.last,
            step = indices.step,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun slice(x: DataBuffer, xi: Int, xj: Int, axis: Int, indices: IntProgression): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> kotlin.math.min(xi, indices.size) * xj
                else -> xi * kotlin.math.min(xj, indices.size)
            },
        )
        shape.sliceD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            axis = axis,
            start = indices.first,
            end = indices.last,
            step = indices.step,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun slice(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int, indices: IntProgression): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> kotlin.math.min(xi, indices.size) * xj * xk
                1 -> xi * kotlin.math.min(xj, indices.size) * xk
                else -> xi * xj * kotlin.math.min(xk, indices.size)
            },
        )
        shape.sliceD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            axis = axis,
            start = indices.first,
            end = indices.last,
            step = indices.step,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, indices: IntProgression) {
        when (y) {
            is GPUJvmBuffer -> {
                shape.copyIntoD1(
                    x = x.toGPUBuffer().ptr,
                    result = y.ptr,
                    start = indices.first,
                    end = indices.last,
                    step = indices.step,
                    runtime = runtime,
                )
            }

            else -> KotlinBackend.copyInto(x = x, y = y, indices = indices)
        }
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int, indices: IntProgression) {
        when (y) {
            is GPUJvmBuffer -> {
                shape.copyIntoD2(
                    x = x.toGPUBuffer().ptr,
                    result = y.ptr,
                    ri = yi,
                    rj = yj,
                    axis = axis,
                    start = indices.first,
                    end = indices.last,
                    step = indices.step,
                    runtime = runtime,
                )
            }

            else -> KotlinBackend.copyInto(x = x, y, indices)
        }
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int, indices: IntProgression) {
        when (y) {
            is GPUJvmBuffer -> {
                shape.copyIntoD3(
                    x = x.toGPUBuffer().ptr,
                    result = y.ptr,
                    ri = yi,
                    rj = yj,
                    rk = yk,
                    axis = axis,
                    start = indices.first,
                    end = indices.last,
                    step = indices.step,
                    runtime = runtime,
                )
            }

            else -> KotlinBackend.copyInto(x = x, y, indices)
        }
    }

    override fun gather(x: DataBuffer, y: DataBuffer, i: Int, j: Int, k: Int): DataBuffer {
        val result = GPUJvmBuffer.create(i * x.size * k)
        index.gather(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            i = i,
            j = j,
            k = k,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun scatterAdd(x: DataBuffer, y: DataBuffer, i: Int, j: Int, k: Int, b: Int): DataBuffer {
        val result = GPUJvmBuffer.create(i * j * k)
        index.scatterAdd(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            i = i,
            j = j,
            k = k,
            b = b,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun greaterThan(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.greaterThanD1ToD0(x = x.toGPUBuffer().ptr, y = y, result = result.ptr, runtime = runtime)
        return result
    }

    override fun greaterThan(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.greaterThanD1ToD1(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun lessThan(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.lessThanD1ToD0(x = x.toGPUBuffer().ptr, y = y, result = result.ptr, runtime = runtime)
        return result
    }

    override fun lessThan(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.lessThanD1ToD1(x = x.toGPUBuffer().ptr, y = y.toGPUBuffer().ptr, result = result.ptr, runtime = runtime)
        return result
    }

    override fun equals(x: DataBuffer, y: Float, absoluteTolerance: Float, relativeTolerance: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.equalsD1ToD0(
            x = x.toGPUBuffer().ptr,
            y = y,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun equals(x: DataBuffer, y: DataBuffer, absoluteTolerance: Float, relativeTolerance: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.equalsD1ToD1(
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun where(condition: DataBuffer, x: Float, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(condition.size)
        jWhere.whereD0ToD0(
            condition = condition.toGPUBuffer().ptr,
            x = x,
            y = y,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun where(condition: DataBuffer, x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        jWhere.whereD0ToD1(
            condition = condition.toGPUBuffer().ptr,
            x = x,
            y = y.toGPUBuffer().ptr,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun where(condition: DataBuffer, x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        jWhere.whereD1ToD0(
            condition = condition.toGPUBuffer().ptr,
            x = x.toGPUBuffer().ptr,
            y = y,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun where(condition: DataBuffer, x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        jWhere.whereD1ToD1(
            condition = condition.toGPUBuffer().ptr,
            x = x.toGPUBuffer().ptr,
            y = y.toGPUBuffer().ptr,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun flip(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        shape.flipD3(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            axis = axis,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun unfold(x: DataBuffer, xi: Int, xj: Int, b: Int, window: Int, stride: Int, padding: Int): DataBuffer {
        val rj = (xj - window + padding * 2) / stride + 1
        val result = GPUJvmBuffer.create(b * xi * rj * window)
        shape.unfoldD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            b = b,
            window = window,
            stride = stride,
            padding = padding,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun unfold(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        b: Int,
        window: Int,
        stride: Int,
        padding: Int,
    ): DataBuffer {
        val rj = (xj - window + padding * 2) / stride + 1
        val rk = (xk - window + padding * 2) / stride + 1
        val result = GPUJvmBuffer.create(b * xi * rj * rk * window * window)
        shape.unfoldD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            b = b,
            window = window,
            stride = stride,
            padding = padding,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun fold(x: DataBuffer, xi: Int, xj: Int, xk: Int, b: Int, stride: Int, padding: Int): DataBuffer {
        val nj = xk + (xj - 1) * stride - padding * 2
        val result = GPUJvmBuffer.create(b * xi * nj)
        shape.foldD1(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            b = b,
            stride = stride,
            padding = padding,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun fold(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        b: Int,
        stride: Int,
        padding: Int,
    ): DataBuffer {
        val window = kotlin.math.sqrt(xl.toDouble()).toInt()
        val nj = window + (xj - 1) * stride - padding * 2
        val nk = window + (xk - 1) * stride - padding * 2
        val result = GPUJvmBuffer.create(b * xi * nj * nk)
        shape.foldD2(
            x = x.toGPUBuffer().ptr,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            b = b,
            stride = stride,
            padding = padding,
            result = result.ptr,
            runtime = runtime,
        )
        return result
    }

    override fun sync() = jRuntime.sync(runtime)

    private fun GPUJvmBuffer.Companion.create(size: Int) = GPUJvmBuffer.create(
        size = size,
        runtime = runtime,
        native = buffer,
    )

    private fun DataBuffer.toGPUBuffer(): GPUJvmBuffer = toGPUBuffer(runtime = runtime, native = buffer)

    companion object {
        private val cleaner = Cleaner.create()
    }
}
