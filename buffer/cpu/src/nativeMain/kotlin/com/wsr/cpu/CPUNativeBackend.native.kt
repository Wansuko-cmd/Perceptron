@file:OptIn(ExperimentalForeignApi::class)

package com.wsr.cpu

import com.wsr.base.IBackend
import com.wsr.base.KotlinBackend
import com.wsr.base.data.DataBuffer
import com.wsr.cpu.rs.com_wsr_cpu_average_d1
import com.wsr.cpu.rs.com_wsr_cpu_average_d2
import com.wsr.cpu.rs.com_wsr_cpu_average_d3
import com.wsr.cpu.rs.com_wsr_cpu_div_d0_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_div_d1_to_d0
import com.wsr.cpu.rs.com_wsr_cpu_div_d1_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_div_d1_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_div_d1_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_div_d2_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_div_d2_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_div_d3_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_div_d3_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_div_d3_to_d4
import com.wsr.cpu.rs.com_wsr_cpu_div_d4_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_div_d4_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_div_d4_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_exp_d1
import com.wsr.cpu.rs.com_wsr_cpu_inner
import com.wsr.cpu.rs.com_wsr_cpu_ln_d1
import com.wsr.cpu.rs.com_wsr_cpu_mat_mul_d1_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_mat_mul_d2_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_mat_mul_d2_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_max_d1
import com.wsr.cpu.rs.com_wsr_cpu_max_d2
import com.wsr.cpu.rs.com_wsr_cpu_max_d3
import com.wsr.cpu.rs.com_wsr_cpu_min_d1
import com.wsr.cpu.rs.com_wsr_cpu_min_d2
import com.wsr.cpu.rs.com_wsr_cpu_min_d3
import com.wsr.cpu.rs.com_wsr_cpu_minus_d0_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_minus_d1_to_d0
import com.wsr.cpu.rs.com_wsr_cpu_minus_d1_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_minus_d1_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_minus_d1_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_minus_d2_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_minus_d2_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_minus_d3_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_minus_d3_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_minus_d3_to_d4
import com.wsr.cpu.rs.com_wsr_cpu_minus_d4_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_minus_d4_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_minus_d4_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_plus_d0_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_plus_d1_to_d0
import com.wsr.cpu.rs.com_wsr_cpu_plus_d1_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_plus_d1_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_plus_d1_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_plus_d2_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_plus_d2_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_plus_d3_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_plus_d3_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_plus_d3_to_d4
import com.wsr.cpu.rs.com_wsr_cpu_plus_d4_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_plus_d4_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_plus_d4_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_pow_d1
import com.wsr.cpu.rs.com_wsr_cpu_sqrt_d1
import com.wsr.cpu.rs.com_wsr_cpu_sum_d1
import com.wsr.cpu.rs.com_wsr_cpu_sum_d2
import com.wsr.cpu.rs.com_wsr_cpu_sum_d3
import com.wsr.cpu.rs.com_wsr_cpu_times_d0_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_times_d1_to_d0
import com.wsr.cpu.rs.com_wsr_cpu_times_d1_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_times_d1_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_times_d1_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_times_d2_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_times_d2_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_times_d3_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_times_d3_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_times_d3_to_d4
import com.wsr.cpu.rs.com_wsr_cpu_times_d4_to_d1
import com.wsr.cpu.rs.com_wsr_cpu_times_d4_to_d2
import com.wsr.cpu.rs.com_wsr_cpu_times_d4_to_d3
import com.wsr.cpu.rs.com_wsr_cpu_transpose_d2
import com.wsr.cpu.rs.com_wsr_cpu_transpose_d3
import com.wsr.cpu.rs.com_wsr_cpu_transpose_d4
import kotlinx.cinterop.ExperimentalForeignApi

actual fun loadCPUBackend(): IBackend? = CPUNativeBackend()

class CPUNativeBackend : IBackend by KotlinBackend {
    override val generator = CPUNativeBuffer.generator

    // 0次元
    override fun plus(x: Float, y: DataBuffer): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_plus_d0_to_d1(x = x, y = y.toCPUBuffer().buffer, y_size = y.size, result = result.buffer)
        return result
    }

    // 1次元
    override fun plus(x: DataBuffer, y: Float): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_plus_d1_to_d0(x = x.toCPUBuffer().buffer, x_size = x.size, y = y, result = result.buffer)
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_plus_d1_to_d1(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            size = x.size,
            result = result.buffer,
        )
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_plus_d1_to_d2(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis = axis,
            result = result.buffer,
        )
        return result
    }

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_plus_d1_to_d3(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis = axis,
            result = result.buffer,
        )
        return result
    }

    // 2次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_plus_d2_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_plus_d2_to_d3(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
        )
        return result
    }

    // 3次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_plus_d3_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_plus_d3_to_d2(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_plus_d3_to_d4(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            yl = yl,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.buffer,
        )
        return result
    }

    // 4次元
    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_plus_d4_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_plus_d4_to_d2(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_plus_d4_to_d3(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.buffer,
        )
        return result
    }

    // 0次元
    override fun minus(x: Float, y: DataBuffer): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_minus_d0_to_d1(x = x, y = y.toCPUBuffer().buffer, y_size = y.size, result = result.buffer)
        return result
    }

    // 1次元
    override fun minus(x: DataBuffer, y: Float): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_minus_d1_to_d0(x = x.toCPUBuffer().buffer, x_size = x.size, y = y, result = result.buffer)
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_minus_d1_to_d1(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            size = x.size,
            result = result.buffer,
        )
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_minus_d1_to_d2(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis = axis,
            result = result.buffer,
        )
        return result
    }

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_minus_d1_to_d3(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis = axis,
            result = result.buffer,
        )
        return result
    }

    // 2次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_minus_d2_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_minus_d2_to_d3(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
        )
        return result
    }

    // 3次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_minus_d3_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_minus_d3_to_d2(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_minus_d3_to_d4(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            yl = yl,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.buffer,
        )
        return result
    }

    // 4次元
    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_minus_d4_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_minus_d4_to_d2(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_minus_d4_to_d3(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.buffer,
        )
        return result
    }

    // 0次元
    override fun times(x: Float, y: DataBuffer): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_times_d0_to_d1(x = x, y = y.toCPUBuffer().buffer, y_size = y.size, result = result.buffer)
        return result
    }

    // 1次元
    override fun times(x: DataBuffer, y: Float): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_times_d1_to_d0(x = x.toCPUBuffer().buffer, x_size = x.size, y = y, result = result.buffer)
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_times_d1_to_d1(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            size = x.size,
            result = result.buffer,
        )
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_times_d1_to_d2(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis = axis,
            result = result.buffer,
        )
        return result
    }

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_times_d1_to_d3(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis = axis,
            result = result.buffer,
        )
        return result
    }

    // 2次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_times_d2_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_times_d2_to_d3(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
        )
        return result
    }

    // 3次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_times_d3_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_times_d3_to_d2(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_times_d3_to_d4(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            yl = yl,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.buffer,
        )
        return result
    }

    // 4次元
    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_times_d4_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_times_d4_to_d2(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_times_d4_to_d3(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.buffer,
        )
        return result
    }

    // 0次元
    override fun div(x: Float, y: DataBuffer): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_div_d0_to_d1(x = x, y = y.toCPUBuffer().buffer, y_size = y.size, result = result.buffer)
        return result
    }

    // 1次元
    override fun div(x: DataBuffer, y: Float): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_div_d1_to_d0(x = x.toCPUBuffer().buffer, x_size = x.size, y = y, result = result.buffer)
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_div_d1_to_d1(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            size = x.size,
            result = result.buffer,
        )
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_div_d1_to_d2(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis = axis,
            result = result.buffer,
        )
        return result
    }

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_div_d1_to_d3(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis = axis,
            result = result.buffer,
        )
        return result
    }

    // 2次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_div_d2_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_div_d2_to_d3(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
        )
        return result
    }

    // 3次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_div_d3_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_div_d3_to_d2(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(y.size)
        com_wsr_cpu_div_d3_to_d4(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            yl = yl,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.buffer,
        )
        return result
    }

    // 4次元
    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_div_d4_to_d1(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            axis = axis,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_div_d4_to_d2(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            axis1 = axis1,
            axis2 = axis2,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_div_d4_to_d3(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            y = y.toCPUBuffer().buffer,
            yi = yi,
            yj = yj,
            yk = yk,
            axis1 = axis1,
            axis2 = axis2,
            axis3 = axis3,
            result = result.buffer,
        )
        return result
    }

    override fun inner(x: DataBuffer, y: DataBuffer, b: Int): DataBuffer {
        val result = CPUNativeBuffer.create(b)
        com_wsr_cpu_inner(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            size = x.size,
            b = b,
            result = result.buffer,
        )
        return result
    }

    override fun matMul(x: DataBuffer, transX: Boolean, y: DataBuffer, m: Int, k: Int): DataBuffer {
        val result = CPUNativeBuffer.create(m)
        com_wsr_cpu_mat_mul_d2_to_d1(
            x = x.toCPUBuffer().buffer,
            trans_x = transX,
            y = y.toCPUBuffer().buffer,
            m = m,
            k = k,
            result = result.buffer,
        )
        return result
    }

    override fun matMul(x: DataBuffer, y: DataBuffer, transY: Boolean, n: Int, k: Int): DataBuffer {
        val result = CPUNativeBuffer.create(n)
        com_wsr_cpu_mat_mul_d1_to_d2(
            x = x.toCPUBuffer().buffer,
            y = y.toCPUBuffer().buffer,
            trans_y = transY,
            n = n,
            k = k,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(b * m * n)
        com_wsr_cpu_mat_mul_d2_to_d2(
            x = x.toCPUBuffer().buffer,
            trans_x = transX,
            y = y.toCPUBuffer().buffer,
            trans_y = transY,
            m = m,
            n = n,
            k = k,
            b = b,
            result = result.buffer,
        )
        return result
    }

    override fun exp(x: DataBuffer): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_exp_d1(x = x.toCPUBuffer().buffer, size = x.size, result = result.buffer)
        return result
    }

    override fun ln(x: DataBuffer, e: Float): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_ln_d1(x = x.toCPUBuffer().buffer, e = e, size = x.size, result = result.buffer)
        return result
    }

    override fun pow(x: DataBuffer, n: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_pow_d1(x = x.toCPUBuffer().buffer, n = n, size = x.size, result = result.buffer)
        return result
    }

    override fun sqrt(x: DataBuffer, e: Float): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_sqrt_d1(x = x.toCPUBuffer().buffer, e = e, size = x.size, result = result.buffer)
        return result
    }

    override fun average(x: DataBuffer): Float = com_wsr_cpu_average_d1(x = x.toCPUBuffer().buffer, size = x.size)

    override fun average(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        com_wsr_cpu_average_d2(x = x.toCPUBuffer().buffer, xi = xi, xj = xj, axis = axis, result = result.buffer)
        return result
    }

    override fun average(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        com_wsr_cpu_average_d3(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            axis = axis,
            result = result.buffer,
        )
        return result
    }

    override fun max(x: DataBuffer): Float = com_wsr_cpu_max_d1(x = x.toCPUBuffer().buffer, size = x.size)

    override fun max(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        com_wsr_cpu_max_d2(x = x.toCPUBuffer().buffer, xi = xi, xj = xj, axis = axis, result = result.buffer)
        return result
    }

    override fun max(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        com_wsr_cpu_max_d3(x = x.toCPUBuffer().buffer, xi = xi, xj = xj, xk = xk, axis = axis, result = result.buffer)
        return result
    }

    override fun min(x: DataBuffer): Float = com_wsr_cpu_min_d1(x = x.toCPUBuffer().buffer, size = x.size)

    override fun min(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        com_wsr_cpu_min_d2(x = x.toCPUBuffer().buffer, xi = xi, xj = xj, axis = axis, result = result.buffer)
        return result
    }

    override fun min(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        com_wsr_cpu_min_d3(x = x.toCPUBuffer().buffer, xi = xi, xj = xj, xk = xk, axis = axis, result = result.buffer)
        return result
    }

    override fun sum(x: DataBuffer): Float = com_wsr_cpu_sum_d1(x = x.toCPUBuffer().buffer, size = x.size)

    override fun sum(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(
            size = when (axis) {
                0 -> xj
                else -> xi
            },
        )
        com_wsr_cpu_sum_d2(x = x.toCPUBuffer().buffer, xi = xi, xj = xj, axis = axis, result = result.buffer)
        return result
    }

    override fun sum(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer {
        val result = CPUNativeBuffer.create(
            size = when (axis) {
                0 -> xj * xk
                1 -> xi * xk
                else -> xi * xj
            },
        )
        com_wsr_cpu_sum_d3(x = x.toCPUBuffer().buffer, xi = xi, xj = xj, xk = xk, axis = axis, result = result.buffer)
        return result
    }

    override fun transpose(x: DataBuffer, xi: Int, xj: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_transpose_d2(x = x.toCPUBuffer().buffer, xi = xi, xj = xj, result = result.buffer)
        return result
    }

    override fun transpose(x: DataBuffer, xi: Int, xj: Int, xk: Int, axisI: Int, axisJ: Int, axisK: Int): DataBuffer {
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_transpose_d3(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            axis_i = axisI,
            axis_j = axisJ,
            axis_k = axisK,
            result = result.buffer,
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
        val result = CPUNativeBuffer.create(x.size)
        com_wsr_cpu_transpose_d4(
            x = x.toCPUBuffer().buffer,
            xi = xi,
            xj = xj,
            xk = xk,
            xl = xl,
            axis_i = axisI,
            axis_j = axisJ,
            axis_k = axisK,
            axis_l = axisL,
            result = result.buffer,
        )
        return result
    }
}
