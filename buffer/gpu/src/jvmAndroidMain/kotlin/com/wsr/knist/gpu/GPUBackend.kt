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
import com.wsr.knist.gpu.reduction.JCollection
import com.wsr.knist.gpu.shape.JShape
import java.lang.ref.Cleaner

class GPUBackend : IBackend by KotlinBackend {
    private val runtime = JRuntime().allocate()
    private val buffer = JBuffer()
    override val generator: IDataBufferGenerator = GPUJvmBuffer.createGenerator(runtime, buffer)

    private val math = JMath()
    private val matMul = JMatMul()
    private val collection = JCollection()
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
        plus.plusD0ToD1(x, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    // 1次元
    override fun plus(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        plus.plusD1ToD0(x.toGPUBuffer().ptr, y, result.ptr, runtime)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        plus.plusD1ToD1(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        plus.plusD1ToD2(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, yi, yj, axis, result.ptr, runtime)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        plus.plusD1ToD3(
            x.toGPUBuffer().ptr,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis,
            result.ptr,
            runtime,
        )
        return result
    }

    // 2次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        plus.plusD2ToD1(x.toGPUBuffer().ptr, xi, xj, y.toGPUBuffer().ptr, axis, result.ptr, runtime)
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
            runtime,
        )
        return result
    }

    // 3次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        plus.plusD3ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            runtime,
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
            runtime,
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
            runtime,
        )
        return result
    }

    // 4次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        plus.plusD4ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            runtime,
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
            runtime,
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
            runtime,
        )
        return result
    }

    // 0次元
    override fun minus(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        minus.minusD0ToD1(x, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    // 1次元
    override fun minus(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        minus.minusD1ToD0(x.toGPUBuffer().ptr, y, result.ptr, runtime)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        minus.minusD1ToD1(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        minus.minusD1ToD2(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, yi, yj, axis, result.ptr, runtime)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        minus.minusD1ToD3(
            x.toGPUBuffer().ptr,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis,
            result.ptr,
            runtime,
        )
        return result
    }

    // 2次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        minus.minusD2ToD1(x.toGPUBuffer().ptr, xi, xj, y.toGPUBuffer().ptr, axis, result.ptr, runtime)
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
            runtime,
        )
        return result
    }

    // 3次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        minus.minusD3ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            runtime,
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
            runtime,
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
            runtime,
        )
        return result
    }

    // 4次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        minus.minusD4ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            runtime,
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
            runtime,
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
            runtime,
        )
        return result
    }

    // 0次元
    override fun times(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        times.timesD0ToD1(x, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    // 1次元
    override fun times(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        times.timesD1ToD0(x.toGPUBuffer().ptr, y, result.ptr, runtime)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        times.timesD1ToD1(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        times.timesD1ToD2(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, yi, yj, axis, result.ptr, runtime)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        times.timesD1ToD3(
            x.toGPUBuffer().ptr,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis,
            result.ptr,
            runtime,
        )
        return result
    }

    // 2次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        times.timesD2ToD1(x.toGPUBuffer().ptr, xi, xj, y.toGPUBuffer().ptr, axis, result.ptr, runtime)
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
            runtime,
        )
        return result
    }

    // 3次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        times.timesD3ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            runtime,
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
            runtime,
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
            runtime,
        )
        return result
    }

    // 4次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        times.timesD4ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            runtime,
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
            runtime,
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
            runtime,
        )
        return result
    }

    // 0次元
    override fun div(x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        div.divD0ToD1(x, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    // 1次元
    override fun div(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        div.divD1ToD0(x.toGPUBuffer().ptr, y, result.ptr, runtime)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        div.divD1ToD1(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        div.divD1ToD2(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, yi, yj, axis, result.ptr, runtime)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        div.divD1ToD3(
            x.toGPUBuffer().ptr,
            y.toGPUBuffer().ptr,
            yi,
            yj,
            yk,
            axis,
            result.ptr,
            runtime,
        )
        return result
    }

    // 2次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        div.divD2ToD1(x.toGPUBuffer().ptr, xi, xj, y.toGPUBuffer().ptr, axis, result.ptr, runtime)
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
            runtime,
        )
        return result
    }

    // 3次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        div.divD3ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            runtime,
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
            runtime,
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
            runtime,
        )
        return result
    }

    // 4次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        div.divD4ToD1(
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            xl,
            y.toGPUBuffer().ptr,
            axis,
            result.ptr,
            runtime,
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
            runtime,
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
            runtime,
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
            runtime,
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
            runtime,
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
            runtime,
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
            runtime,
        )
        return result
    }

    override fun exp(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.exp(x.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun ln(x: DataBuffer, e: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.ln(x.toGPUBuffer().ptr, e, result.ptr, runtime)
        return result
    }

    override fun sigmoid(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.sigmoid(x.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun pow(x: DataBuffer, n: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.pow(x.toGPUBuffer().ptr, n, result.ptr, runtime)
        return result
    }

    override fun sqrt(x: DataBuffer, e: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        math.sqrt(x.toGPUBuffer().ptr, e, result.ptr, runtime)
        return result
    }

    override fun average(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        collection.averageD1(x.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun average(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.averageD2(x.toGPUBuffer().ptr, xi, xj, axis, result.ptr, runtime)
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
        collection.averageD3(x.toGPUBuffer().ptr, xi, xj, xk, axis, result.ptr, runtime)
        return result
    }

    override fun max(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        collection.maxD1(x.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun max(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.maxD2(x.toGPUBuffer().ptr, xi, xj, axis, result.ptr, runtime)
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
        collection.maxD3(x.toGPUBuffer().ptr, xi, xj, xk, axis, result.ptr, runtime)
        return result
    }

    override fun min(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        collection.minD1(x.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun min(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.minD2(x.toGPUBuffer().ptr, xi, xj, axis, result.ptr, runtime)
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
        collection.minD3(x.toGPUBuffer().ptr, xi, xj, xk, axis, result.ptr, runtime)
        return result
    }

    override fun sum(x: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(1)
        collection.sumD1(x.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun sum(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = GPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.sumD2(x.toGPUBuffer().ptr, xi, xj, axis, result.ptr, runtime)
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
        collection.sumD3(x.toGPUBuffer().ptr, xi, xj, xk, axis, result.ptr, runtime)
        return result
    }

    override fun transpose(x: DataBuffer, xi: Int, xj: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        shape.transposeD2(x.toGPUBuffer().ptr, xi, xj, result.ptr, runtime)
        return result
    }

    override fun transpose(x: DataBuffer, xi: Int, xj: Int, xk: Int, axisI: Int, axisJ: Int, axisK: Int): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        shape.transposeD3(x.toGPUBuffer().ptr, xi, xj, xk, axisI, axisJ, axisK, result.ptr, runtime)
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
        shape.transposeD4(x.toGPUBuffer().ptr, xi, xj, xk, xl, axisI, axisJ, axisK, axisL, result.ptr, runtime)
        return result
    }

    override fun slice(x: DataBuffer, indices: IntProgression): DataBuffer {
        val result = GPUJvmBuffer.create(kotlin.math.min(x.size, indices.size))
        shape.sliceD1(x.toGPUBuffer().ptr, indices.first, indices.last, indices.step, result.ptr, runtime)
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
            x.toGPUBuffer().ptr,
            xi,
            xj,
            axis,
            indices.first,
            indices.last,
            indices.step,
            result.ptr,
            runtime,
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
            x.toGPUBuffer().ptr,
            xi,
            xj,
            xk,
            axis,
            indices.first,
            indices.last,
            indices.step,
            result.ptr,
            runtime,
        )
        return result
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, indices: IntProgression) {
        when (y) {
            is GPUJvmBuffer -> {
                shape.copyIntoD1(
                    x.toGPUBuffer().ptr,
                    y.ptr,
                    indices.first,
                    indices.last,
                    indices.step,
                    runtime,
                )
            }
            else -> KotlinBackend.copyInto(x, y, indices)
        }
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int, indices: IntProgression) {
        when (y) {
            is GPUJvmBuffer -> {
                shape.copyIntoD2(
                    x.toGPUBuffer().ptr,
                    y.ptr,
                    yi,
                    yj,
                    axis,
                    indices.first,
                    indices.last,
                    indices.step,
                    runtime,
                )
            }
            else -> KotlinBackend.copyInto(x, y, indices)
        }
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int, indices: IntProgression) {
        when (y) {
            is GPUJvmBuffer -> {
                shape.copyIntoD3(
                    x.toGPUBuffer().ptr,
                    y.ptr,
                    yi,
                    yj,
                    yk,
                    axis,
                    indices.first,
                    indices.last,
                    indices.step,
                    runtime,
                )
            }
            else -> KotlinBackend.copyInto(x, y, indices)
        }
    }

    override fun gather(x: DataBuffer, y: DataBuffer, i: Int, j: Int, k: Int): DataBuffer {
        val result = GPUJvmBuffer.create(i * x.size * k)
        index.gather(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, i, j, k, result.ptr, runtime)
        return result
    }

    override fun scatterAdd(x: DataBuffer, y: DataBuffer, i: Int, j: Int, k: Int, b: Int): DataBuffer {
        val result = GPUJvmBuffer.create(i * j * k)
        index.scatterAdd(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, i, j, k, b, result.ptr, runtime)
        return result
    }

    override fun greaterThan(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.greaterThanD1ToD0(x.toGPUBuffer().ptr, y, result.ptr, runtime)
        return result
    }

    override fun greaterThan(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.greaterThanD1ToD1(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun lessThan(x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.lessThanD1ToD0(x.toGPUBuffer().ptr, y, result.ptr, runtime)
        return result
    }

    override fun lessThan(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.lessThanD1ToD1(x.toGPUBuffer().ptr, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun equals(x: DataBuffer, y: Float, absoluteTolerance: Float, relativeTolerance: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.equalsD1ToD0(x.toGPUBuffer().ptr, y, absoluteTolerance, relativeTolerance, result.ptr, runtime)
        return result
    }

    override fun equals(x: DataBuffer, y: DataBuffer, absoluteTolerance: Float, relativeTolerance: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        compare.equalsD1ToD1(
            x.toGPUBuffer().ptr,
            y.toGPUBuffer().ptr,
            absoluteTolerance,
            relativeTolerance,
            result.ptr,
            runtime,
        )
        return result
    }

    override fun where(condition: DataBuffer, x: Float, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(condition.size)
        jWhere.whereD0ToD0(condition.toGPUBuffer().ptr, x, y, result.ptr, runtime)
        return result
    }

    override fun where(condition: DataBuffer, x: Float, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(y.size)
        jWhere.whereD0ToD1(condition.toGPUBuffer().ptr, x, y.toGPUBuffer().ptr, result.ptr, runtime)
        return result
    }

    override fun where(condition: DataBuffer, x: DataBuffer, y: Float): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        jWhere.whereD1ToD0(condition.toGPUBuffer().ptr, x.toGPUBuffer().ptr, y, result.ptr, runtime)
        return result
    }

    override fun where(condition: DataBuffer, x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = GPUJvmBuffer.create(x.size)
        jWhere.whereD1ToD1(
            condition.toGPUBuffer().ptr,
            x.toGPUBuffer().ptr,
            y.toGPUBuffer().ptr,
            result.ptr,
            runtime,
        )
        return result
    }

    private fun GPUJvmBuffer.Companion.create(size: Int) = GPUJvmBuffer.create(size, runtime, buffer)

    private fun DataBuffer.toGPUBuffer(): GPUJvmBuffer = toGPUBuffer(runtime, buffer)

    companion object {
        private val cleaner = Cleaner.create()
    }
}
