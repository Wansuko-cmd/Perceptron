package com.wsr.cpu

import com.wsr.base.IBackend
import com.wsr.base.KotlinBackend
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.size
import com.wsr.base.loadNativeLibrary
import com.wsr.cpu.elementwise.compare.JCompare
import com.wsr.cpu.elementwise.compare.where.JWhere
import com.wsr.cpu.elementwise.math.JMath
import com.wsr.cpu.elementwise.operation.div.JDiv
import com.wsr.cpu.elementwise.operation.minus.JMinus
import com.wsr.cpu.elementwise.operation.plus.JPlus
import com.wsr.cpu.elementwise.operation.times.JTimes
import com.wsr.cpu.index.JIndex
import com.wsr.cpu.linalg.JMatMul
import com.wsr.cpu.reduction.JCollection
import com.wsr.cpu.shape.JShape
import kotlin.math.min

private const val LIB_PATH = "cpu"
private const val LIB_NAME = "cpu"

actual fun loadCPUBackend(): IBackend? {
    val isSuccess = loadNativeLibrary(path = LIB_PATH, name = LIB_NAME)
    return if (isSuccess) CPUJvmBackend() else null
}

class CPUJvmBackend : IBackend by KotlinBackend {
    private val collection = JCollection()
    private val compare = JCompare()
    private val where = JWhere()
    private val index = JIndex()
    private val math = JMath()
    private val matMul = JMatMul()
    private val plus = JPlus()
    private val minus = JMinus()
    private val times = JTimes()
    private val div = JDiv()
    private val shape = JShape()

    override val generator = CPUJvmBuffer.generator

    // 0次元
    override fun plus(x: Float, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        plus.plusD0ToD1(x, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    // 1次元
    override fun plus(x: DataBuffer, y: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        plus.plusD1ToD0(x.toCPUBuffer().byteBuffer, y, result.byteBuffer)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        plus.plusD1ToD1(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        plus.plusD1ToD2(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, yi, yj, axis, result.byteBuffer)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        plus.plusD1ToD3(
            x.toCPUBuffer().byteBuffer,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis,
            result.byteBuffer,
        )
        return result
    }

    // 2次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        plus.plusD2ToD1(x.toCPUBuffer().byteBuffer, xi, xj, y.toCPUBuffer().byteBuffer, axis, result.byteBuffer)
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
        val result = CPUJvmBuffer.create(y.size)
        plus.plusD2ToD3(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            result.byteBuffer,
        )
        return result
    }

    // 3次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        plus.plusD3ToD1(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            axis,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        plus.plusD3ToD2(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            axis1,
            axis2,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(y.size)
        plus.plusD3ToD4(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            yl,
            axis1,
            axis2,
            axis3,
            result.byteBuffer,
        )
        return result
    }

    // 4次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        plus.plusD4ToD1(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            axis,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        plus.plusD4ToD2(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            axis1,
            axis2,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        plus.plusD4ToD3(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            axis3,
            result.byteBuffer,
        )
        return result
    }

    // 0次元
    override fun minus(x: Float, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        minus.minusD0ToD1(x, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    // 1次元
    override fun minus(x: DataBuffer, y: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        minus.minusD1ToD0(x.toCPUBuffer().byteBuffer, y, result.byteBuffer)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        minus.minusD1ToD1(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        minus.minusD1ToD2(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, yi, yj, axis, result.byteBuffer)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        minus.minusD1ToD3(
            x.toCPUBuffer().byteBuffer,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis,
            result.byteBuffer,
        )
        return result
    }

    // 2次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        minus.minusD2ToD1(x.toCPUBuffer().byteBuffer, xi, xj, y.toCPUBuffer().byteBuffer, axis, result.byteBuffer)
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
        val result = CPUJvmBuffer.create(y.size)
        minus.minusD2ToD3(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            result.byteBuffer,
        )
        return result
    }

    // 3次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        minus.minusD3ToD1(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            axis,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        minus.minusD3ToD2(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            axis1,
            axis2,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(y.size)
        minus.minusD3ToD4(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            yl,
            axis1,
            axis2,
            axis3,
            result.byteBuffer,
        )
        return result
    }

    // 4次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        minus.minusD4ToD1(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            axis,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        minus.minusD4ToD2(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            axis1,
            axis2,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        minus.minusD4ToD3(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            axis3,
            result.byteBuffer,
        )
        return result
    }

    // 0次元
    override fun times(x: Float, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        times.timesD0ToD1(x, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    // 1次元
    override fun times(x: DataBuffer, y: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        times.timesD1ToD0(x.toCPUBuffer().byteBuffer, y, result.byteBuffer)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        times.timesD1ToD1(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        times.timesD1ToD2(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, yi, yj, axis, result.byteBuffer)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        times.timesD1ToD3(
            x.toCPUBuffer().byteBuffer,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis,
            result.byteBuffer,
        )
        return result
    }

    // 2次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        times.timesD2ToD1(x.toCPUBuffer().byteBuffer, xi, xj, y.toCPUBuffer().byteBuffer, axis, result.byteBuffer)
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
        val result = CPUJvmBuffer.create(y.size)
        times.timesD2ToD3(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            result.byteBuffer,
        )
        return result
    }

    // 3次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        times.timesD3ToD1(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            axis,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        times.timesD3ToD2(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            axis1,
            axis2,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(y.size)
        times.timesD3ToD4(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            yl,
            axis1,
            axis2,
            axis3,
            result.byteBuffer,
        )
        return result
    }

    // 4次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        times.timesD4ToD1(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            axis,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        times.timesD4ToD2(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            axis1,
            axis2,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        times.timesD4ToD3(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            axis3,
            result.byteBuffer,
        )
        return result
    }

    // 0次元
    override fun div(x: Float, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        div.divD0ToD1(x, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    // 1次元
    override fun div(x: DataBuffer, y: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        div.divD1ToD0(x.toCPUBuffer().byteBuffer, y, result.byteBuffer)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        div.divD1ToD1(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        div.divD1ToD2(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, yi, yj, axis, result.byteBuffer)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        div.divD1ToD3(
            x.toCPUBuffer().byteBuffer,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis,
            result.byteBuffer,
        )
        return result
    }

    // 2次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        div.divD2ToD1(x.toCPUBuffer().byteBuffer, xi, xj, y.toCPUBuffer().byteBuffer, axis, result.byteBuffer)
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
        val result = CPUJvmBuffer.create(y.size)
        div.divD2ToD3(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            result.byteBuffer,
        )
        return result
    }

    // 3次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        div.divD3ToD1(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            axis,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        div.divD3ToD2(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            axis1,
            axis2,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(y.size)
        div.divD3ToD4(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            yl,
            axis1,
            axis2,
            axis3,
            result.byteBuffer,
        )
        return result
    }

    // 4次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        div.divD4ToD1(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            axis,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        div.divD4ToD2(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            axis1,
            axis2,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(x.size)
        div.divD4ToD3(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            y.toCPUBuffer().byteBuffer,
            yi,
            yj,
            yk,
            axis1,
            axis2,
            axis3,
            result.byteBuffer,
        )
        return result
    }

    override fun inner(x: DataBuffer, y: DataBuffer, b: Int): DataBuffer {
        val result = CPUJvmBuffer.create(b)
        matMul.inner(
            x.toCPUBuffer().byteBuffer,
            y.toCPUBuffer().byteBuffer,
            b,
            result.byteBuffer,
        )
        return result
    }

    override fun matMul(x: DataBuffer, transX: Boolean, y: DataBuffer, m: Int, k: Int): DataBuffer {
        val result = CPUJvmBuffer.create(m)
        matMul.matMulD2ToD1(
            x.toCPUBuffer().byteBuffer,
            transX,
            y.toCPUBuffer().byteBuffer,
            m,
            k,
            result.byteBuffer,
        )
        return result
    }

    override fun matMul(x: DataBuffer, y: DataBuffer, transY: Boolean, n: Int, k: Int): DataBuffer {
        val result = CPUJvmBuffer.create(n)
        matMul.matMulD1ToD2(
            x.toCPUBuffer().byteBuffer,
            y.toCPUBuffer().byteBuffer,
            transY,
            n,
            k,
            result.byteBuffer,
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
        val result = CPUJvmBuffer.create(b * m * n)
        matMul.matMulD2ToD2(
            x.toCPUBuffer().byteBuffer,
            transX,
            y.toCPUBuffer().byteBuffer,
            transY,
            m,
            n,
            k,
            b,
            result.byteBuffer,
        )
        return result
    }

    override fun exp(x: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        math.exp(x.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    override fun ln(x: DataBuffer, e: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        math.ln(x.toCPUBuffer().byteBuffer, e, result.byteBuffer)
        return result
    }

    override fun sigmoid(x: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        math.sigmoid(x.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    override fun pow(x: DataBuffer, n: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        math.pow(x.toCPUBuffer().byteBuffer, n, result.byteBuffer)
        return result
    }

    override fun sqrt(x: DataBuffer, e: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        math.sqrt(x.toCPUBuffer().byteBuffer, e, result.byteBuffer)
        return result
    }

    override fun average(x: DataBuffer): DataBuffer {
        val result = collection.averageD1(x.toCPUBuffer().byteBuffer)
        return CPUJvmBuffer.create(floatArrayOf(result))
    }

    override fun average(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.averageD2(x.toCPUBuffer().byteBuffer, xi, xj, axis, result.byteBuffer)
        return result
    }

    override fun average(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        collection.averageD3(x.toCPUBuffer().byteBuffer, xi, xj, xk, axis, result.byteBuffer)
        return result
    }

    override fun max(x: DataBuffer): DataBuffer {
        val result = collection.maxD1(x.toCPUBuffer().byteBuffer)
        return CPUJvmBuffer.create(floatArrayOf(result))
    }

    override fun max(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.maxD2(x.toCPUBuffer().byteBuffer, xi, xj, axis, result.byteBuffer)
        return result
    }

    override fun max(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        collection.maxD3(x.toCPUBuffer().byteBuffer, xi, xj, xk, axis, result.byteBuffer)
        return result
    }

    override fun min(x: DataBuffer): DataBuffer {
        val result = collection.minD1(x.toCPUBuffer().byteBuffer)
        return CPUJvmBuffer.create(floatArrayOf(result))
    }

    override fun min(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.minD2(x.toCPUBuffer().byteBuffer, xi, xj, axis, result.byteBuffer)
        return result
    }

    override fun min(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        collection.minD3(x.toCPUBuffer().byteBuffer, xi, xj, xk, axis, result.byteBuffer)
        return result
    }

    override fun sum(x: DataBuffer): DataBuffer {
        val result = collection.sumD1(x.toCPUBuffer().byteBuffer)
        return CPUJvmBuffer.create(floatArrayOf(result))
    }

    override fun sum(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        collection.sumD2(x.toCPUBuffer().byteBuffer, xi, xj, axis, result.byteBuffer)
        return result
    }

    override fun sum(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = CPUJvmBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        collection.sumD3(x.toCPUBuffer().byteBuffer, xi, xj, xk, axis, result.byteBuffer)
        return result
    }

    override fun transpose(x: DataBuffer, xi: Int, xj: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        shape.transposeD2(x.toCPUBuffer().byteBuffer, xi, xj, result.byteBuffer)
        return result
    }

    override fun transpose(x: DataBuffer, xi: Int, xj: Int, xk: Int, axisI: Int, axisJ: Int, axisK: Int): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        shape.transposeD3(x.toCPUBuffer().byteBuffer, xi, xj, xk, axisI, axisJ, axisK, result.byteBuffer)
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
        val result = CPUJvmBuffer.create(x.size)
        shape.transposeD4(x.toCPUBuffer().byteBuffer, xi, xj, xk, xl, axisI, axisJ, axisK, axisL, result.byteBuffer)
        return result
    }

    override fun transpose(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        xm: Int,
        axisI: Int,
        axisJ: Int,
        axisK: Int,
        axisL: Int,
        axisM: Int,
    ): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        shape.transposeD5(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            xl,
            xm,
            axisI,
            axisJ,
            axisK,
            axisL,
            axisM,
            result.byteBuffer,
        )
        return result
    }

    override fun slice(x: DataBuffer, indices: IntProgression): DataBuffer {
        val result = CPUJvmBuffer.create(min(x.size, indices.size))
        shape.sliceD1(x.toCPUBuffer().byteBuffer, indices.first, indices.last, indices.step, result.byteBuffer)
        return result
    }

    override fun slice(x: DataBuffer, xi: Int, xj: Int, axis: Int, indices: IntProgression): DataBuffer {
        val result = CPUJvmBuffer.create(
            size = when (axis) {
                0 -> min(xi, indices.size) * xj
                else -> xi * min(xj, indices.size)
            },
        )
        shape.sliceD2(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            axis,
            indices.first,
            indices.last,
            indices.step,
            result.byteBuffer,
        )
        return result
    }

    override fun slice(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int, indices: IntProgression): DataBuffer {
        val result = CPUJvmBuffer.create(
            size = when (axis) {
                0 -> min(xi, indices.size) * xj * xk
                1 -> xi * min(xj, indices.size) * xk
                else -> xi * xj * min(xk, indices.size)
            },
        )
        shape.sliceD3(
            x.toCPUBuffer().byteBuffer,
            xi,
            xj,
            xk,
            axis,
            indices.first,
            indices.last,
            indices.step,
            result.byteBuffer,
        )
        return result
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, indices: IntProgression) {
        when (y) {
            is CPUJvmBuffer -> {
                shape.copyIntoD1(
                    x.toCPUBuffer().byteBuffer,
                    y.byteBuffer,
                    indices.first,
                    indices.last,
                    indices.step,
                )
            }
            else -> KotlinBackend.copyInto(x, y, indices)
        }
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int, indices: IntProgression) {
        when (y) {
            is CPUJvmBuffer -> {
                shape.copyIntoD2(
                    x.toCPUBuffer().byteBuffer,
                    y.byteBuffer,
                    yi,
                    yj,
                    axis,
                    indices.first,
                    indices.last,
                    indices.step,
                )
            }
            else -> KotlinBackend.copyInto(x, y, indices)
        }
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int, indices: IntProgression) {
        when (y) {
            is CPUJvmBuffer -> {
                shape.copyIntoD3(
                    x.toCPUBuffer().byteBuffer,
                    y.byteBuffer,
                    yi,
                    yj,
                    yk,
                    axis,
                    indices.first,
                    indices.last,
                    indices.step,
                )
            }
            else -> KotlinBackend.copyInto(x, y, indices)
        }
    }

    override fun gather(x: DataBuffer, y: DataBuffer, i: Int, j: Int, k: Int): DataBuffer {
        val result = CPUJvmBuffer.create(i * x.size * k)
        index.gather(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, i, j, k, result.byteBuffer)
        return result
    }

    override fun scatterAdd(x: DataBuffer, y: DataBuffer, i: Int, j: Int, k: Int, b: Int): DataBuffer {
        val result = CPUJvmBuffer.create(i * j * k)
        index.scatterAdd(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, i, j, k, b, result.byteBuffer)
        return result
    }

    override fun greaterThan(x: DataBuffer, y: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        compare.greaterThanD1ToD0(x.toCPUBuffer().byteBuffer, y, result.byteBuffer)
        return result
    }

    override fun greaterThan(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        compare.greaterThanD1ToD1(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    override fun lessThan(x: DataBuffer, y: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        compare.lessThanD1ToD0(x.toCPUBuffer().byteBuffer, y, result.byteBuffer)
        return result
    }

    override fun lessThan(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        compare.lessThanD1ToD1(x.toCPUBuffer().byteBuffer, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    override fun equals(x: DataBuffer, y: Float, absoluteTolerance: Float, relativeTolerance: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        compare.equalsD1ToD0(x.toCPUBuffer().byteBuffer, y, absoluteTolerance, relativeTolerance, result.byteBuffer)
        return result
    }

    override fun equals(x: DataBuffer, y: DataBuffer, absoluteTolerance: Float, relativeTolerance: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        compare.equalsD1ToD1(
            x.toCPUBuffer().byteBuffer,
            y.toCPUBuffer().byteBuffer,
            absoluteTolerance,
            relativeTolerance,
            result.byteBuffer,
        )
        return result
    }

    override fun where(condition: DataBuffer, x: Float, y: Float): DataBuffer {
        val result = CPUJvmBuffer.create(condition.size)
        where.whereD0ToD0(condition.toCPUBuffer().byteBuffer, x, y, result.byteBuffer)
        return result
    }

    override fun where(condition: DataBuffer, x: Float, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(y.size)
        where.whereD0ToD1(condition.toCPUBuffer().byteBuffer, x, y.toCPUBuffer().byteBuffer, result.byteBuffer)
        return result
    }

    override fun where(condition: DataBuffer, x: DataBuffer, y: Float): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        where.whereD1ToD0(condition.toCPUBuffer().byteBuffer, x.toCPUBuffer().byteBuffer, y, result.byteBuffer)
        return result
    }

    override fun where(condition: DataBuffer, x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUJvmBuffer.create(x.size)
        where.whereD1ToD1(
            condition.toCPUBuffer().byteBuffer,
            x.toCPUBuffer().byteBuffer,
            y.toCPUBuffer().byteBuffer,
            result.byteBuffer,
        )
        return result
    }

    override fun unfold(x: DataBuffer, xi: Int, xj: Int, b: Int, window: Int, stride: Int, padding: Int): DataBuffer {
        val oj = (xj + padding * 2 - window) / stride + 1
        val result = CPUJvmBuffer.create(b * xi * oj * window)
        shape.unfold(x.toCPUBuffer().byteBuffer, xi, xj, b, window, stride, padding, result.byteBuffer)
        return result
    }

    override fun fold(x: DataBuffer, xi: Int, xj: Int, xk: Int, b: Int, stride: Int, padding: Int): DataBuffer {
        val oj = xk + (xj - 1) * stride - padding * 2
        val result = CPUJvmBuffer.create(b * xi * oj)
        shape.fold(x.toCPUBuffer().byteBuffer, xi, xj, xk, b, stride, padding, result.byteBuffer)
        return result
    }
}
