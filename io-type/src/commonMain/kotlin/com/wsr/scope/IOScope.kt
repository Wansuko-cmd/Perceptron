@file:Suppress("NOTHING_TO_INLINE")

package com.wsr.scope

import com.wsr.Backend
import com.wsr.BufferScope
import com.wsr.base.data.DataBuffer
import com.wsr.base.data.create
import com.wsr.base.data.indices
import com.wsr.base.data.size
import com.wsr.batch.Batch
import com.wsr.core.IOType
import com.wsr.core.compare.equals.EQUALS_ABSOLUTE_TOLERANCE
import com.wsr.core.compare.equals.EQUALS_RELATIVE_TOLERANCE
import kotlin.jvm.JvmName
import kotlin.random.Random

class IOScope(private val scope: BufferScope = BufferScope()) : AutoCloseable {
    fun register(buffer: DataBuffer) {
        scope.register(buffer)
    }

    fun remove(buffer: DataBuffer) {
        scope.remove(buffer)
    }

    override fun close() {
        scope.close()
    }

    @JvmName("26235")
    inline fun IOType.D1.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))

    @JvmName("30310")
    inline fun IOType.D1.reshapeToD2(shape: List<Int>) =
        IOType.D2(shape = shape, value = value).also { register(it.value) }

    @JvmName("19114")
    inline fun IOType.D2.reshapeToD3(i: Int, j: Int, k: Int) = reshapeToD3(shape = listOf(i, j, k))

    @JvmName("16931")
    inline fun IOType.D2.reshapeToD3(shape: List<Int>) =
        IOType.D3(shape = shape, value = value).also { register(it.value) }

    @JvmName("25694")
    inline fun IOType.D2.reshapeToD4(i: Int, j: Int, k: Int, l: Int) =
        IOType.D4(shape = listOf(i, j, k, l), value = value).also { register(it.value) }

    @JvmName("24300")
    inline fun IOType.D3.reshapeToD2(i: Int, j: Int) =
        IOType.D2(shape = listOf(i, j), value = value).also { register(it.value) }

    @JvmName("3123")
    inline fun IOType.D4.reshapeToD2(i: Int, j: Int) =
        IOType.D2(shape = listOf(i, j), value = value).also { register(it.value) }

    @JvmName("6974")
    inline fun IOType.D4.reshapeToD3(i: Int, j: Int, k: Int) =
        IOType.D3(shape = listOf(i, j, k), value = value).also { register(it.value) }

    @JvmName("6164")
    inline fun IOType.D1.slice(indices: IntProgression): IOType.D1 {
        val result = Backend.slice(x = value, indices = indices)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("20118")
    inline fun IOType.D2.slice(indices: IntProgression, axis: Int): IOType.D2 {
        val result = Backend.slice(x = value, xi = i, xj = j, axis = axis, indices = indices)
        return IOType.D2(
            shape = when (axis) {
                0 -> listOf(indices.size, j)
                else -> listOf(i, indices.size)
            },
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("13362")
    inline fun IOType.D2.transpose(): IOType.D2 {
        val result = Backend.transpose(x = value, xi = i, xj = j)
        return IOType.D2(shape = listOf(j, i), value = result).also { register(it.value) }
    }

    @JvmName("22604")
    inline fun IOType.D3.transpose(axisI: Int, axisJ: Int, axisK: Int): IOType.D3 {
        val result = Backend.transpose(x = value, xi = i, xj = j, xk = k, axisI = axisI, axisJ = axisJ, axisK = axisK)
        return IOType.D3(shape = listOf(shape[axisI], shape[axisJ], shape[axisK]), value = result)
            .also { register(it.value) }
    }

    @JvmName("5791")
    inline fun IOType.D4.transpose(axisI: Int, axisJ: Int, axisK: Int, axisL: Int): IOType.D4 {
        val axes = listOf(axisI, axisJ, axisK, axisL)
        return IOType.d4(i = shape[axisI], j = shape[axisJ], k = shape[axisK], l = shape[axisL]) { i, j, k, l ->
            val indices = listOf(i, j, k, l)
            this[
                indices[axes.indexOf(0)],
                indices[axes.indexOf(1)],
                indices[axes.indexOf(2)],
                indices[axes.indexOf(3)],
            ]
        }.also { register(it.value) }
    }

    @JvmName("29398")
    inline fun IOType.D1.broadcastToD2(axis: Int, size: Int) = when (axis) {
        0 -> IOType.d2(size, shape[0]) { x, y -> this[y] }.also { register(it.value) }
        1 -> IOType.d2(shape[0], size) { x, y -> this[x] }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D1.broadcastToD2 axis is $axis not 0 or 1.")
    }

    @JvmName("18935")
    inline fun IOType.D2.broadcastToD3(axis: Int, size: Int) = when (axis) {
        0 -> IOType.d3(size, shape[0], shape[1]) { i, j, k -> this[j, k] }.also { register(it.value) }
        1 -> IOType.d3(shape[0], size, shape[1]) { i, j, k -> this[i, k] }.also { register(it.value) }
        2 -> IOType.d3(shape[0], shape[1], size) { i, j, k -> this[i, j] }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D2.broadcastToD3 axis is $axis not 0, 1 or 2.")
    }

    @JvmName("1329")
    inline fun IOType.D1.interleave(other: IOType.D1): IOType.D1 {
        check(size == other.size)

        val result = DataBuffer.create(size + other.size)
        Backend.copyInto(x = value, y = result, indices = 0 until size * 2 step 2)
        Backend.copyInto(x = other.value, y = result, indices = 1 until size * 2 step 2)

        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("6224")
    inline fun IOType.D2.interleave(other: IOType.D2, axis: Int): IOType.D2 {
        check(i == other.i && j == other.j)
        val result = DataBuffer.create(size + other.size)
        val newShape = when (axis) {
            0 -> listOf(i * 2, j)
            else -> listOf(i, j * 2)
        }
        val (i, j) = newShape
        Backend.copyInto(
            x = value,
            y = result,
            yi = i,
            yj = j,
            axis = axis,
            indices = when (axis) {
                0 -> 0 until i step 2
                else -> 0 until j step 2
            },
        )
        Backend.copyInto(
            x = other.value,
            y = result,
            yi = i,
            yj = j,
            axis = axis,
            indices = when (axis) {
                0 -> 1 until i step 2
                else -> 1 until j step 2
            },
        )
        return IOType.D2(shape = newShape, value = result).also { register(it.value) }
    }

    @JvmName("12656")
    inline operator fun IOType.D4.div(other: Float): IOType.D4 {
        val result = Backend.div(x = value, y = other)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("23674")
    inline operator fun IOType.D4.div(other: IOType.D4): IOType.D4 {
        val result = Backend.div(x = value, y = other.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5944")
    inline operator fun IOType.D2.div(other: Float): IOType.D2 {
        val result = Backend.div(x = value, y = other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("19024")
    inline operator fun IOType.D2.div(other: IOType.D0): IOType.D2 {
        val result = Backend.div(x = value, y = other.get())
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13125")
    inline fun IOType.D2.div(other: IOType.D1, axis: Int): IOType.D2 {
        val result = Backend.div(x = value, xi = i, xj = j, y = other.value, axis = axis)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3266")
    inline operator fun IOType.D2.div(other: IOType.D2): IOType.D2 {
        val result = Backend.div(x = value, y = other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("31606")
    inline fun IOType.D2.div(other: IOType.D3, axis1: Int, axis2: Int): IOType.D3 {
        val result = Backend.div(
            x = value,
            xi = i,
            xj = j,
            y = other.value,
            yi = other.i,
            yj = other.j,
            yk = other.k,
            axis1 = axis1,
            axis2 = axis2,
        )
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("5308")
    inline operator fun Float.div(other: IOType.D0): IOType.D0 =
        IOType.d0(this / other.get()).also { register(it.value) }

    @JvmName("16206")
    inline operator fun Float.div(other: IOType.D1): IOType.D1 {
        val result = Backend.div(x = this, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("5956")
    inline operator fun Float.div(other: IOType.D2): IOType.D2 {
        val result = Backend.div(x = this, y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("29347")
    inline operator fun Float.div(other: IOType.D3): IOType.D3 {
        val result = Backend.div(x = this, y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("29647")
    inline operator fun Float.div(other: IOType.D4): IOType.D4 {
        val result = Backend.div(x = this, y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("2217")
    inline operator fun IOType.D0.div(other: Float): IOType.D0 = IOType.d0(get() / other).also { register(it.value) }

    @JvmName("4721")
    inline operator fun IOType.D0.div(other: IOType.D0): IOType.D0 =
        IOType.d0(get() / other.get()).also { register(it.value) }

    @JvmName("11950")
    inline operator fun IOType.D0.div(other: IOType.D1): IOType.D1 {
        val result = Backend.div(x = get(), y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("21693")
    inline operator fun IOType.D0.div(other: IOType.D2): IOType.D2 {
        val result = Backend.div(x = get(), y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("4966")
    inline operator fun IOType.D0.div(other: IOType.D3): IOType.D3 {
        val result = Backend.div(x = get(), y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("15793")
    inline operator fun IOType.D0.div(other: IOType.D4): IOType.D4 {
        val result = Backend.div(x = get(), y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("19422")
    inline operator fun IOType.D1.div(other: Float): IOType.D1 {
        val result = Backend.div(x = value, y = other)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("25963")
    inline operator fun IOType.D1.div(other: IOType.D0): IOType.D1 {
        val result = Backend.div(x = value, y = other.get())
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("32165")
    inline operator fun IOType.D1.div(other: IOType.D1): IOType.D1 {
        val result = Backend.div(x = value, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("22248")
    inline fun IOType.D1.div(other: IOType.D2, axis: Int): IOType.D2 {
        val result = Backend.div(x = value, y = other.value, yi = other.i, yj = other.j, axis = axis)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("24243")
    inline operator fun IOType.D3.div(other: Float): IOType.D3 {
        val result = Backend.div(x = value, y = other)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("1088")
    inline operator fun IOType.D3.div(other: IOType.D0): IOType.D3 {
        val result = Backend.div(x = value, y = other.get())
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("23869")
    inline fun IOType.D3.div(other: IOType.D1, axis: Int): IOType.D3 {
        val result = Backend.div(
            x = value,
            xi = i,
            xj = j,
            xk = k,
            y = other.value,
            axis = axis,
        )
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("21077")
    inline fun IOType.D3.div(other: IOType.D2, axis1: Int, axis2: Int): IOType.D3 {
        val result = Backend.div(
            x = value,
            xi = i,
            xj = j,
            xk = k,
            y = other.value,
            yi = other.i,
            yj = other.j,
            axis1 = axis1,
            axis2 = axis2,
        )
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("235")
    inline operator fun IOType.D3.div(other: IOType.D3): IOType.D3 {
        val result = Backend.div(x = value, y = other.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("31586")
    inline operator fun IOType.D4.minus(other: Float): IOType.D4 {
        val result = Backend.minus(x = value, y = other)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5113")
    inline operator fun IOType.D4.minus(other: IOType.D4): IOType.D4 {
        val result = Backend.minus(x = value, y = other.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("20822")
    inline operator fun IOType.D2.minus(other: Float): IOType.D2 {
        val result = Backend.minus(x = value, y = other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5116")
    inline operator fun IOType.D2.minus(other: IOType.D0): IOType.D2 {
        val result = Backend.minus(x = value, y = other.get())
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("28306")
    inline fun IOType.D2.minus(other: IOType.D1, axis: Int): IOType.D2 {
        val result = Backend.minus(x = value, xi = i, xj = j, y = other.value, axis = axis)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("752")
    inline operator fun IOType.D2.minus(other: IOType.D2): IOType.D2 {
        val result = Backend.minus(x = value, y = other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("1445")
    inline fun IOType.D2.minus(other: IOType.D3, axis1: Int, axis2: Int): IOType.D3 {
        val result = Backend.minus(
            x = value,
            xi = i,
            xj = j,
            y = other.value,
            yi = other.i,
            yj = other.j,
            yk = other.k,
            axis1 = axis1,
            axis2 = axis2,
        )
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("7786")
    inline operator fun Float.minus(other: IOType.D0): IOType.D0 =
        IOType.d0(value = this - other.get()).also { register(it.value) }

    @JvmName("28702")
    inline operator fun Float.minus(other: IOType.D1): IOType.D1 {
        val result = Backend.minus(x = this, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("2967")
    inline operator fun Float.minus(other: IOType.D2): IOType.D2 {
        val result = Backend.minus(x = this, y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("22724")
    inline operator fun Float.minus(other: IOType.D3): IOType.D3 {
        val result = Backend.minus(x = this, y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("24747")
    inline operator fun Float.minus(other: IOType.D4): IOType.D4 {
        val result = Backend.minus(x = this, y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("6882")
    inline operator fun IOType.D0.minus(other: Float): IOType.D0 =
        IOType.d0(value = get() - other).also { register(it.value) }

    @JvmName("22621")
    inline operator fun IOType.D0.minus(other: IOType.D0): IOType.D0 =
        IOType.d0(get() - other.get()).also { register(it.value) }

    @JvmName("8350")
    inline operator fun IOType.D0.minus(other: IOType.D1): IOType.D1 {
        val result = Backend.minus(x = get(), y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("31761")
    inline operator fun IOType.D0.minus(other: IOType.D2): IOType.D2 {
        val result = Backend.minus(x = get(), y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("416")
    inline operator fun IOType.D0.minus(other: IOType.D3): IOType.D3 {
        val result = Backend.minus(x = get(), y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("30031")
    inline operator fun IOType.D0.minus(other: IOType.D4): IOType.D4 {
        val result = Backend.minus(x = get(), y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("16326")
    inline operator fun IOType.D1.minus(other: Float): IOType.D1 {
        val result = Backend.minus(x = value, y = other)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("17219")
    inline operator fun IOType.D1.minus(other: IOType.D0): IOType.D1 {
        val result = Backend.minus(x = value, y = other.get())
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("14359")
    inline operator fun IOType.D1.minus(other: IOType.D1): IOType.D1 {
        val result = Backend.minus(x = value, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("23587")
    inline fun IOType.D1.minus(other: IOType.D2, axis: Int): IOType.D2 {
        val result = Backend.minus(x = value, y = other.value, yi = other.i, yj = other.j, axis = axis)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("9592")
    inline operator fun IOType.D3.minus(other: Float): IOType.D3 {
        val result = Backend.minus(x = value, y = other)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("26367")
    inline operator fun IOType.D3.minus(other: IOType.D0): IOType.D3 {
        val result = Backend.minus(x = value, y = other.get())
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("7894")
    inline fun IOType.D3.minus(other: IOType.D1, axis: Int): IOType.D3 {
        val result = Backend.minus(
            x = value,
            xi = i,
            xj = j,
            xk = k,
            y = other.value,
            axis = axis,
        )
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("11001")
    inline fun IOType.D3.minus(other: IOType.D2, axis1: Int, axis2: Int): IOType.D3 {
        val result = Backend.minus(
            x = value,
            xi = i,
            xj = j,
            xk = k,
            y = other.value,
            yi = other.i,
            yj = other.j,
            axis1 = axis1,
            axis2 = axis2,
        )
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("25841")
    inline operator fun IOType.D3.minus(other: IOType.D3): IOType.D3 {
        val result = Backend.minus(x = value, y = other.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("11346")
    inline infix fun IOType.D1.inner(other: IOType.D1): Float = Backend.inner(x = value, y = other.value, b = 1)[0]

    @JvmName("4149")
    inline operator fun IOType.D4.times(other: Float): IOType.D4 {
        val result = Backend.times(x = value, y = other)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("161")
    inline operator fun IOType.D4.times(other: IOType.D4): IOType.D4 {
        val result = Backend.times(x = value, y = other.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("27992")
    inline operator fun IOType.D2.times(other: Float): IOType.D2 {
        val result = Backend.times(x = value, y = other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("32030")
    inline operator fun IOType.D2.times(other: IOType.D0): IOType.D2 {
        val result = Backend.times(x = value, y = other.get())
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("9142")
    inline fun IOType.D2.times(other: IOType.D1, axis: Int): IOType.D2 {
        val result = Backend.times(x = value, xi = i, xj = j, y = other.value, axis = axis)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("16647")
    inline operator fun IOType.D2.times(other: IOType.D2): IOType.D2 {
        val result = Backend.times(x = value, y = other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3603")
    inline fun IOType.D2.times(other: IOType.D3, axis1: Int, axis2: Int): IOType.D3 {
        val result = Backend.times(
            x = value,
            xi = i,
            xj = j,
            y = other.value,
            yi = other.i,
            yj = other.j,
            yk = other.k,
            axis1 = axis1,
            axis2 = axis2,
        )
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("26752")
    inline operator fun Float.times(other: IOType.D0): IOType.D0 =
        IOType.d0(this * other.get()).also { register(it.value) }

    @JvmName("22877")
    inline operator fun Float.times(other: IOType.D1): IOType.D1 {
        val result = Backend.times(x = this, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("31732")
    inline operator fun Float.times(other: IOType.D2): IOType.D2 {
        val result = Backend.times(x = this, y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("26642")
    inline operator fun Float.times(other: IOType.D3): IOType.D3 {
        val result = Backend.times(x = this, y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("5643")
    inline operator fun Float.times(other: IOType.D4): IOType.D4 {
        val result = Backend.times(x = this, y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("25395")
    inline operator fun IOType.D0.times(other: Float): IOType.D0 = IOType.d0(get() * other).also { register(it.value) }

    @JvmName("14898")
    inline operator fun IOType.D0.times(other: IOType.D0): IOType.D0 =
        IOType.d0(get() * other.get()).also { register(it.value) }

    @JvmName("29624")
    inline operator fun IOType.D0.times(other: IOType.D1): IOType.D1 {
        val result = Backend.times(x = get(), y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("3215")
    inline operator fun IOType.D0.times(other: IOType.D2): IOType.D2 {
        val result = Backend.times(x = get(), y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("4858")
    inline operator fun IOType.D0.times(other: IOType.D3): IOType.D3 {
        val result = Backend.times(x = get(), y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("14469")
    inline operator fun IOType.D0.times(other: IOType.D4): IOType.D4 {
        val result = Backend.times(x = get(), y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("11496")
    inline operator fun IOType.D1.times(other: Float): IOType.D1 {
        val result = Backend.times(x = value, y = other)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("30797")
    inline operator fun IOType.D1.times(other: IOType.D0): IOType.D1 {
        val result = Backend.times(x = value, y = other.get())
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("9812")
    inline operator fun IOType.D1.times(other: IOType.D1): IOType.D1 {
        val result = Backend.times(x = value, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("9231")
    inline fun IOType.D1.times(other: IOType.D2, axis: Int): IOType.D2 {
        val result = Backend.times(x = value, y = other.value, yi = other.i, yj = other.j, axis = axis)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("24841")
    inline operator fun IOType.D3.times(other: Float): IOType.D3 {
        val result = Backend.times(x = value, y = other)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("797")
    inline operator fun IOType.D3.times(other: IOType.D0): IOType.D3 {
        val result = Backend.times(x = value, y = other.get())
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("16032")
    inline fun IOType.D3.times(other: IOType.D1, axis: Int): IOType.D3 {
        val result = Backend.times(
            x = value,
            xi = i,
            xj = j,
            xk = k,
            y = other.value,
            axis = axis,
        )
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("2639")
    inline fun IOType.D3.times(other: IOType.D2, axis1: Int, axis2: Int): IOType.D3 {
        val result = Backend.times(
            x = value,
            xi = i,
            xj = j,
            xk = k,
            y = other.value,
            yi = other.i,
            yj = other.j,
            axis1 = axis1,
            axis2 = axis2,
        )
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("11636")
    inline operator fun IOType.D3.times(other: IOType.D3): IOType.D3 {
        val result = Backend.times(x = value, y = other.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("971")
    inline fun IOType.D2.zipWith(other: IOType.D1, axis: Int, block: (Float, Float) -> Float): IOType.D2 = when (axis) {
        0 -> IOType.d2(shape) { i, j -> block(this[i, j], other[i]) }.also { register(it.value) }
        1 -> IOType.d2(shape) { i, j -> block(this[i, j], other[j]) }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D2.zipWith axis is $axis not 0 or 1.")
    }

    @JvmName("25138")
    inline fun IOType.D2.zipWith(other: IOType.D3, axis1: Int, axis2: Int, block: (Float, Float) -> Float): IOType.D3 =
        when (axis1) {
            0 -> when (axis2) {
                1 -> IOType.d3(other.shape) { i, j, k -> block(this[i, j], other[i, j, k]) }.also { register(it.value) }
                2 -> IOType.d3(other.shape) { i, j, k -> block(this[i, k], other[i, j, k]) }.also { register(it.value) }
                else -> throw IllegalArgumentException("IOType.D2.zipWith axis2 is $axis2 not 1 or 2.")
            }

            1 -> when (axis2) {
                2 -> IOType.d3(other.shape) { i, j, k -> block(this[j, k], other[i, j, k]) }.also { register(it.value) }
                else -> throw IllegalArgumentException("IOType.D2.zipWith axis2 is $axis2 not 2.")
            }

            else -> throw IllegalArgumentException("IOType.D2.zipWith axis1 is $axis1 not 0 or 1.")
        }

    @JvmName("30648")
    inline fun IOType.D1.zipWith(other: IOType.D2, axis: Int, block: (Float, Float) -> Float): IOType.D2 = when (axis) {
        0 -> IOType.d2(other.shape) { i, j -> block(this[i], other[i, j]) }.also { register(it.value) }
        1 -> IOType.d2(other.shape) { i, j -> block(this[j], other[i, j]) }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D1.zipWith axis is $axis not 0 or 1.")
    }

    @JvmName("26415")
    inline fun IOType.D1.zipWith(other: IOType.D3, axis: Int, block: (Float, Float) -> Float): IOType.D3 = when (axis) {
        0 -> IOType.d3(shape) { i, j, k -> block(this[i], other[i, j, k]) }.also { register(it.value) }
        1 -> IOType.d3(shape) { i, j, k -> block(this[j], other[i, j, k]) }.also { register(it.value) }
        2 -> IOType.d3(shape) { i, j, k -> block(this[k], other[i, j, k]) }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D3.zipWith axis is $axis not 0, 1 or 2.")
    }

    @JvmName("24823")
    inline fun IOType.D3.zipWith(other: IOType.D1, axis: Int, block: (Float, Float) -> Float): IOType.D3 = when (axis) {
        0 -> IOType.d3(shape) { i, j, k -> block(this[i, j, k], other[i]) }.also { register(it.value) }
        1 -> IOType.d3(shape) { i, j, k -> block(this[i, j, k], other[j]) }.also { register(it.value) }
        2 -> IOType.d3(shape) { i, j, k -> block(this[i, j, k], other[k]) }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D3.zipWith axis is $axis not 0, 1 or 2.")
    }

    @JvmName("2129")
    inline fun IOType.D3.zipWith(other: IOType.D2, axis1: Int, axis2: Int, block: (Float, Float) -> Float): IOType.D3 =
        when (axis1) {
            0 -> when (axis2) {
                1 -> IOType.d3(shape) { i, j, k -> block(this[i, j, k], other[i, j]) }.also { register(it.value) }
                2 -> IOType.d3(shape) { i, j, k -> block(this[i, j, k], other[i, k]) }.also { register(it.value) }
                else -> throw IllegalArgumentException("IOType.D3.zipWith axis2 is $axis2 not 1 or 2.")
            }

            1 -> when (axis2) {
                2 -> IOType.d3(shape) { i, j, k -> block(this[i, j, k], other[j, k]) }.also { register(it.value) }
                else -> throw IllegalArgumentException("IOType.D3.zipWith axis2 is $axis2 not 2.")
            }

            else -> throw IllegalArgumentException("IOType.D3.zipWith axis1 is $axis1 not 0 or 1.")
        }

    @JvmName("20474")
    inline fun IOType.D2.matMul(other: IOType.D1, trans: Boolean = false): IOType.D1 {
        val m = if (trans) shape[1] else shape[0]
        val k = if (trans) shape[0] else shape[1]
        val result = Backend.matMul(
            x = value,
            transX = trans,
            y = other.value,
            m = m,
            k = k,
        )
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("13934")
    inline fun IOType.D2.matMul(other: IOType.D2, transA: Boolean = false, transB: Boolean = false): IOType.D2 {
        val m = if (transA) shape[1] else shape[0]
        val n = if (transB) other.shape[0] else other.shape[1]
        val k = if (transA) shape[0] else shape[1]
        val result = Backend.matMul(
            x = value,
            transX = transA,
            y = other.value,
            transY = transB,
            m = m,
            n = n,
            k = k,
            b = 1,
        )
        return IOType.D2(result, listOf(m, n)).also { register(it.value) }
    }

    @JvmName("20707")
    inline fun IOType.D3.matMul(other: IOType.D3, transA: Boolean = false, transB: Boolean = false): IOType.D3 {
        val m = if (transA) shape[2] else shape[1]
        val n = if (transB) other.shape[1] else other.shape[2]
        val k = if (transA) shape[1] else shape[2]
        val result = Backend.matMul(
            x = value,
            transX = transA,
            y = other.value,
            transY = transB,
            m = m,
            n = n,
            k = k,
            b = shape[0],
        )
        return IOType.D3(shape = listOf(shape[0], m, n), value = result).also { register(it.value) }
    }

    @JvmName("11682")
    inline operator fun IOType.D4.plus(other: Float): IOType.D4 {
        val result = Backend.plus(x = value, y = other)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("4679")
    inline operator fun IOType.D4.plus(other: IOType.D4): IOType.D4 {
        val result = Backend.plus(x = value, y = other.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("8776")
    inline operator fun IOType.D2.plus(other: Float): IOType.D2 {
        val result = Backend.plus(x = value, y = other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5426")
    inline operator fun IOType.D2.plus(other: IOType.D0): IOType.D2 {
        val result = Backend.plus(x = value, y = other.get())
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("7874")
    inline fun IOType.D2.plus(other: IOType.D1, axis: Int): IOType.D2 {
        val result = Backend.plus(x = value, xi = i, xj = j, y = other.value, axis = axis)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("26766")
    inline operator fun IOType.D2.plus(other: IOType.D2): IOType.D2 {
        val result = Backend.plus(x = value, y = other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("25945")
    inline fun IOType.D2.plus(other: IOType.D3, axis1: Int, axis2: Int): IOType.D3 {
        val result = Backend.plus(
            x = value,
            xi = i,
            xj = j,
            y = other.value,
            yi = other.i,
            yj = other.j,
            yk = other.k,
            axis1 = axis1,
            axis2 = axis2,
        )
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("18377")
    inline operator fun Float.plus(other: IOType.D0): IOType.D0 =
        IOType.d0(value = this + other.get()).also { register(it.value) }

    @JvmName("10573")
    inline operator fun Float.plus(other: IOType.D1): IOType.D1 {
        val result = Backend.plus(x = this, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("2275")
    inline operator fun Float.plus(other: IOType.D2): IOType.D2 {
        val result = Backend.plus(x = this, y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("29725")
    inline operator fun Float.plus(other: IOType.D3): IOType.D3 {
        val result = Backend.plus(x = this, y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("4535")
    inline operator fun Float.plus(other: IOType.D4): IOType.D4 {
        val result = Backend.plus(x = this, y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("32685")
    inline operator fun IOType.D0.plus(other: Float): IOType.D0 =
        IOType.d0(value = get() + other).also { register(it.value) }

    @JvmName("13306")
    inline operator fun IOType.D0.plus(other: IOType.D0): IOType.D0 =
        IOType.d0(get() + other.get()).also { register(it.value) }

    @JvmName("25287")
    inline operator fun IOType.D0.plus(other: IOType.D1): IOType.D1 {
        val result = Backend.plus(x = get(), y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("1322")
    inline operator fun IOType.D0.plus(other: IOType.D2): IOType.D2 {
        val result = Backend.plus(x = get(), y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("32706")
    inline operator fun IOType.D0.plus(other: IOType.D3): IOType.D3 {
        val result = Backend.plus(x = get(), y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("15624")
    inline operator fun IOType.D0.plus(other: IOType.D4): IOType.D4 {
        val result = Backend.plus(x = get(), y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("16063")
    inline operator fun IOType.D1.plus(other: Float): IOType.D1 {
        val result = Backend.plus(x = value, y = other)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("10965")
    inline operator fun IOType.D1.plus(other: IOType.D0): IOType.D1 {
        val result = Backend.plus(x = value, y = other.get())
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("14364")
    inline operator fun IOType.D1.plus(other: IOType.D1): IOType.D1 {
        val result = Backend.plus(x = value, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("17508")
    inline fun IOType.D1.plus(other: IOType.D2, axis: Int): IOType.D2 {
        val result = Backend.plus(x = value, y = other.value, yi = other.i, yj = other.j, axis = axis)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("20613")
    inline operator fun IOType.D3.plus(other: Float): IOType.D3 {
        val result = Backend.plus(x = value, y = other)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("31580")
    inline operator fun IOType.D3.plus(other: IOType.D0): IOType.D3 {
        val result = Backend.plus(x = value, y = other.get())
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3002")
    inline fun IOType.D3.plus(other: IOType.D1, axis: Int): IOType.D3 {
        val result = Backend.plus(
            x = value,
            xi = i,
            xj = j,
            xk = k,
            y = other.value,
            axis = axis,
        )
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5687")
    inline fun IOType.D3.plus(other: IOType.D2, axis1: Int, axis2: Int): IOType.D3 {
        val result = Backend.plus(
            x = value,
            xi = i,
            xj = j,
            xk = k,
            y = other.value,
            yi = other.i,
            yj = other.j,
            axis1 = axis1,
            axis2 = axis2,
        )
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("20800")
    inline operator fun IOType.D3.plus(other: IOType.D3): IOType.D3 {
        val result = Backend.plus(x = value, y = other.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("7491")
    inline fun IOType.D1.max(): IOType.D0 = IOType.D0(Backend.max(x = value)).also { register(it.value) }

    @JvmName("10171")
    inline fun IOType.D2.max(): IOType.D0 = IOType.D0(Backend.max(x = value)).also { register(it.value) }

    @JvmName("31642")
    inline fun IOType.D2.max(axis: Int): IOType.D1 {
        val result = Backend.max(x = value, xi = i, xj = j, axis = axis)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("15038")
    inline fun IOType.D3.max(): IOType.D0 = IOType.D0(Backend.max(x = value)).also { register(it.value) }

    @JvmName("2682")
    inline fun IOType.D3.max(axis: Int): IOType.D2 {
        val result = Backend.max(x = value, xi = i, xj = j, xk = k, axis = axis)
        return IOType.D2(
            shape = when (axis) {
                0 -> listOf(j, k)
                1 -> listOf(i, k)
                else -> listOf(i, j)
            },
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("24836")
    inline fun IOType.D1.sum(): IOType.D0 = IOType.D0(Backend.sum(x = value)).also { register(it.value) }

    @JvmName("2951")
    inline fun IOType.D2.sum(): IOType.D0 = IOType.D0(Backend.sum(x = value)).also { register(it.value) }

    @JvmName("16059")
    inline fun IOType.D2.sum(axis: Int): IOType.D1 {
        val result = Backend.sum(x = value, xi = i, xj = j, axis = axis)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("20465")
    inline fun IOType.D3.sum(): IOType.D0 = IOType.D0(Backend.sum(value)).also { register(it.value) }

    @JvmName("11664")
    inline fun IOType.D3.sum(axis: Int): IOType.D2 {
        val result = Backend.sum(x = value, xi = i, xj = j, xk = k, axis = axis)
        return IOType.D2(
            shape = when (axis) {
                0 -> listOf(j, k)
                1 -> listOf(i, k)
                else -> listOf(i, j)
            },
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("18777")
    inline fun IOType.D4.sum(): IOType.D0 = IOType.D0(Backend.sum(value)).also { register(it.value) }

    @JvmName("16585")
    inline fun IOType.D1.topK(k: Int, random: Random = Random): Int = Backend.topK(x = value, k = k, random = random)

    @JvmName("14641")
    inline fun IOType.D1.topP(p: Float, random: Random = Random): Int = Backend.topP(x = value, p = p, random = random)

    @JvmName("28159")
    inline fun IOType.D1.maxIndex(): Int = Backend.maxIndex(x = value)

    @JvmName("20067")
    inline fun IOType.D1.min() = IOType.D0(Backend.min(x = value)).also { register(it.value) }

    @JvmName("25417")
    inline fun IOType.D2.min() = IOType.D0(Backend.min(x = value)).also { register(it.value) }

    @JvmName("32201")
    inline fun IOType.D2.min(axis: Int): IOType.D1 {
        val result = Backend.min(x = value, xi = i, xj = j, axis = axis)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("5127")
    inline fun IOType.D3.min() = IOType.D0(Backend.min(x = value)).also { register(it.value) }

    @JvmName("16381")
    inline fun IOType.D3.min(axis: Int): IOType.D2 {
        val result = Backend.min(x = value, xi = i, xj = j, xk = k, axis = axis)
        return IOType.D2(
            shape = when (axis) {
                0 -> listOf(j, k)
                1 -> listOf(i, k)
                else -> listOf(i, j)
            },
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("8288")
    inline fun IOType.D1.average(): IOType.D0 = IOType.D0(Backend.average(value)).also { register(it.value) }

    @JvmName("15352")
    inline fun IOType.D2.average(): IOType.D0 = IOType.D0(Backend.average(value)).also { register(it.value) }

    @JvmName("6366")
    inline fun IOType.D2.average(axis: Int): IOType.D1 {
        val result = Backend.average(x = value, xi = i, xj = j, axis = axis)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("12403")
    inline fun IOType.D3.average(): IOType.D0 = IOType.D0(Backend.average(value)).also { register(it.value) }

    @JvmName("5039")
    inline fun IOType.D3.average(axis: Int): IOType.D2 {
        val result = Backend.average(x = value, xi = i, xj = j, xk = k, axis = axis)
        return IOType.D2(
            shape = when (axis) {
                0 -> listOf(j, k)
                1 -> listOf(i, k)
                else -> listOf(i, j)
            },
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("15029")
    inline fun IOType.D1.ln(e: Float): IOType.D1 {
        val result = Backend.ln(x = value, e = e)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("5768")
    inline fun IOType.D2.ln(e: Float): IOType.D2 {
        val result = Backend.ln(x = value, e = e)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("0")
    inline fun IOType.D3.ln(e: Float): IOType.D3 {
        val result = Backend.ln(x = value, e = e)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("28811")
    inline fun IOType.D1.softmax(): IOType.D1 {
        val max = max()
        val exp = (this - max).exp()
        val sum = exp.sum()
        return (exp / sum).also { register(it.value) }
    }

    @JvmName("4663")
    inline fun IOType.D2.softmax(): IOType.D2 {
        val max = max()
        val exp = (this - max).exp()
        val sum = exp.sum()
        return (exp / sum).also { register(it.value) }
    }

    @JvmName("22560")
    inline fun IOType.D2.softmax(axis: Int): IOType.D2 {
        val max = max(axis = axis)
        val exp = this.minus(other = max, axis = if (axis == 0) 1 else 0).exp()
        val sum = exp.sum(axis = axis)
        return exp.div(other = sum, axis = if (axis == 0) 1 else 0).also { register(it.value) }
    }

    @JvmName("32104")
    inline fun IOType.D3.softmax(): IOType.D3 {
        val max = max()
        val exp = (this - max).exp()
        val sum = exp.sum()
        return (exp / sum).also { register(it.value) }
    }

    @JvmName("25262")
    inline fun IOType.D3.softmax(axis: Int): IOType.D3 {
        val axis1 = when (axis) {
            0 -> 1
            else -> 0
        }
        val axis2 = when (axis) {
            0, 1 -> 2
            else -> 1
        }
        val max = max(axis = axis)
        val exp = this.minus(other = max, axis1 = axis1, axis2 = axis2).exp()
        val sum = exp.sum(axis = axis)
        return exp.div(other = sum, axis1 = axis1, axis2 = axis2).also { register(it.value) }
    }

    @JvmName("20509")
    inline fun IOType.D1.exp(): IOType.D1 {
        val result = Backend.exp(x = value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("24543")
    inline fun IOType.D2.exp(): IOType.D2 {
        val result = Backend.exp(x = value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("1463")
    inline fun IOType.D3.exp(): IOType.D3 {
        val result = Backend.exp(x = value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("11316")
    inline fun IOType.D0.pow(n: Int): IOType.D0 {
        val result = Backend.pow(x = value, n = n)
        return IOType.D0(value = result).also { register(it.value) }
    }

    @JvmName("11157")
    inline fun IOType.D1.pow(n: Int): IOType.D1 {
        val result = Backend.pow(x = value, n = n)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("27370")
    inline fun IOType.D2.pow(n: Int): IOType.D2 {
        val result = Backend.pow(x = value, n = n)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("818")
    inline fun IOType.D3.pow(n: Int): IOType.D3 {
        val result = Backend.pow(x = value, n = n)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("31872")
    inline fun IOType.D4.pow(n: Int): IOType.D4 {
        val result = Backend.pow(x = value, n = n)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("15605")
    inline fun IOType.D0.sqrt(e: Float = 1e-7f): IOType.D0 {
        val result = Backend.sqrt(x = value, e = e)
        return IOType.D0(value = result).also { register(it.value) }
    }

    @JvmName("17156")
    inline fun IOType.D1.sqrt(e: Float = 1e-7f): IOType.D1 {
        val result = Backend.sqrt(x = value, e = e)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("14438")
    inline fun IOType.D2.sqrt(e: Float = 1e-7f): IOType.D2 {
        val result = Backend.sqrt(x = value, e = e)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("16548")
    inline fun IOType.D3.sqrt(e: Float = 1e-7f): IOType.D3 {
        val result = Backend.sqrt(x = value, e = e)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3614")
    inline fun IOType.D4.sqrt(e: Float = 1e-7f): IOType.D4 {
        val result = Backend.sqrt(x = value, e = e)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("12803")
    inline infix fun IOType.D0.gt(other: Float): IOType.D0 {
        val result = Backend.greaterThan(value, other)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("14210")
    inline infix fun IOType.D0.gt(other: IOType.D0): IOType.D0 {
        val result = Backend.greaterThan(value, other.value)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("22608")
    inline infix fun IOType.D1.gt(other: Float): IOType.D1 {
        val result = Backend.greaterThan(value, other)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("26671")
    inline infix fun IOType.D1.gt(other: IOType.D1): IOType.D1 {
        val result = Backend.greaterThan(value, other.value)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("26391")
    inline infix fun IOType.D2.gt(other: Float): IOType.D2 {
        val result = Backend.greaterThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("27956")
    inline infix fun IOType.D2.gt(other: IOType.D2): IOType.D2 {
        val result = Backend.greaterThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("18900")
    inline infix fun IOType.D3.gt(other: Float): IOType.D2 {
        val result = Backend.greaterThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("9729")
    inline infix fun IOType.D3.gt(other: IOType.D2): IOType.D2 {
        val result = Backend.greaterThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13644")
    inline infix fun IOType.D4.gt(other: Float): IOType.D2 {
        val result = Backend.greaterThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("27425")
    inline infix fun IOType.D4.gt(other: IOType.D2): IOType.D2 {
        val result = Backend.greaterThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("14491")
    inline infix fun IOType.D0.lt(other: Float): IOType.D0 {
        val result = Backend.lessThan(value, other)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("30377")
    inline infix fun IOType.D0.lt(other: IOType.D0): IOType.D0 {
        val result = Backend.lessThan(value, other.value)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("16908")
    inline infix fun IOType.D1.lt(other: Float): IOType.D1 {
        val result = Backend.lessThan(value, other)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("8815")
    inline infix fun IOType.D1.lt(other: IOType.D1): IOType.D1 {
        val result = Backend.lessThan(value, other.value)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("19371")
    inline infix fun IOType.D2.lt(other: Float): IOType.D2 {
        val result = Backend.lessThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("4090")
    inline infix fun IOType.D2.lt(other: IOType.D2): IOType.D2 {
        val result = Backend.lessThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("9115")
    inline infix fun IOType.D3.lt(other: Float): IOType.D2 {
        val result = Backend.lessThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3643")
    inline infix fun IOType.D3.lt(other: IOType.D2): IOType.D2 {
        val result = Backend.lessThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("1063")
    inline infix fun IOType.D4.lt(other: Float): IOType.D2 {
        val result = Backend.lessThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("20736")
    inline infix fun IOType.D4.lt(other: IOType.D2): IOType.D2 {
        val result = Backend.lessThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("28569")
    inline infix fun IOType.D0.eq(other: Float) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("4368")
    inline fun IOType.D0.eq(
        other: Float,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): IOType.D0 {
        val result = Backend.equals(
            x = value,
            y = other,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("15519")
    inline infix fun IOType.D0.eq(other: IOType.D0) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("28229")
    inline fun IOType.D0.eq(
        other: IOType.D0,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): IOType.D0 {
        val result = Backend.equals(
            x = value,
            y = other.value,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("26894")
    inline infix fun IOType.D1.eq(other: Float) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("31735")
    inline fun IOType.D1.eq(
        other: Float,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): IOType.D1 {
        val result = Backend.equals(
            x = value,
            y = other,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("14306")
    inline infix fun IOType.D1.eq(other: IOType.D1) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("209")
    inline fun IOType.D1.eq(
        other: IOType.D1,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): IOType.D1 {
        val result = Backend.equals(
            x = value,
            y = other.value,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("5698")
    inline infix fun IOType.D2.eq(other: Float) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("13027")
    inline fun IOType.D2.eq(
        other: Float,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): IOType.D2 {
        val result = Backend.equals(
            x = value,
            y = other,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5811")
    inline infix fun IOType.D2.eq(other: IOType.D2) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("3")
    inline fun IOType.D2.eq(
        other: IOType.D2,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): IOType.D2 {
        val result = Backend.equals(
            x = value,
            y = other.value,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("18494")
    inline infix fun IOType.D3.eq(other: Float) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("14074")
    inline fun IOType.D3.eq(
        other: Float,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): IOType.D3 {
        val result = Backend.equals(
            x = value,
            y = other,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("32668")
    inline infix fun IOType.D3.eq(other: IOType.D3) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("6054")
    inline fun IOType.D3.eq(
        other: IOType.D3,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): IOType.D3 {
        val result = Backend.equals(
            x = value,
            y = other.value,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("14881")
    inline infix fun IOType.D4.eq(other: Float) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("26895")
    inline fun IOType.D4.eq(
        other: Float,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): IOType.D4 {
        val result = Backend.equals(
            x = value,
            y = other,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13821")
    inline infix fun IOType.D4.eq(other: IOType.D4) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("14072")
    inline fun IOType.D4.eq(
        other: IOType.D4,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): IOType.D4 {
        val result = Backend.equals(
            x = value,
            y = other.value,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("8466")
    inline fun where(condition: IOType.D4, onTrue: Float, onFalse: Float): IOType.D4 {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return IOType.D4(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("81")
    inline fun where(condition: IOType.D4, onTrue: Float, onFalse: IOType.D4): IOType.D4 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D4(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("26142")
    inline fun where(condition: IOType.D4, onTrue: IOType.D4, onFalse: Float): IOType.D4 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D4(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("10825")
    inline fun where(condition: IOType.D4, onTrue: IOType.D4, onFalse: IOType.D4): IOType.D4 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D4(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("46")
    inline fun IOType.D4.where(onTrue: Float, onFalse: Float, condition: (IOType.D4) -> IOType.D4) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("27816")
    inline fun IOType.D4.where(condition: IOType.D4, onTrue: Float, onFalse: IOType.D4 = this): IOType.D4 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("24894")
    inline fun IOType.D4.where(onTrue: Float, onFalse: IOType.D4 = this, condition: (IOType.D4) -> IOType.D4) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("25591")
    inline fun IOType.D4.where(condition: IOType.D4, onTrue: IOType.D4 = this, onFalse: Float): IOType.D4 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("4176")
    inline fun IOType.D4.where(onTrue: IOType.D4 = this, onFalse: Float, condition: (IOType.D4) -> IOType.D4) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("20687")
    inline fun IOType.D4.where(condition: IOType.D4, onTrue: IOType.D4 = this, onFalse: IOType.D4 = this): IOType.D4 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("4731")
    inline fun IOType.D4.where(
        onTrue: IOType.D4 = this,
        onFalse: IOType.D4 = this,
        condition: (IOType.D4) -> IOType.D4,
    ) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("4023")
    inline fun where(condition: IOType.D2, onTrue: Float, onFalse: Float): IOType.D2 {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return IOType.D2(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("29013")
    inline fun where(condition: IOType.D2, onTrue: Float, onFalse: IOType.D2): IOType.D2 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D2(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("438")
    inline fun where(condition: IOType.D2, onTrue: IOType.D2, onFalse: Float): IOType.D2 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D2(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("16822")
    inline fun where(condition: IOType.D2, onTrue: IOType.D2, onFalse: IOType.D2): IOType.D2 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D2(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("20405")
    inline fun IOType.D2.where(onTrue: Float, onFalse: Float, condition: (IOType.D2) -> IOType.D2) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("5703")
    inline fun IOType.D2.where(condition: IOType.D2, onTrue: Float, onFalse: IOType.D2 = this): IOType.D2 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("28646")
    inline fun IOType.D2.where(onTrue: Float, onFalse: IOType.D2 = this, condition: (IOType.D2) -> IOType.D2) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("27788")
    inline fun IOType.D2.where(condition: IOType.D2, onTrue: IOType.D2 = this, onFalse: Float): IOType.D2 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("14838")
    inline fun IOType.D2.where(onTrue: IOType.D2 = this, onFalse: Float, condition: (IOType.D2) -> IOType.D2) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("3092")
    inline fun IOType.D2.where(condition: IOType.D2, onTrue: IOType.D2 = this, onFalse: IOType.D2 = this): IOType.D2 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("929")
    inline fun IOType.D2.where(
        onTrue: IOType.D2 = this,
        onFalse: IOType.D2 = this,
        condition: (IOType.D2) -> IOType.D2,
    ) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("29117")
    inline fun where(condition: IOType.D0, onTrue: Float, onFalse: Float): IOType.D0 {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("26277")
    inline fun where(condition: IOType.D0, onTrue: Float, onFalse: IOType.D0): IOType.D0 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("18361")
    inline fun where(condition: IOType.D0, onTrue: IOType.D0, onFalse: Float): IOType.D0 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("20700")
    inline fun where(condition: IOType.D0, onTrue: IOType.D0, onFalse: IOType.D0): IOType.D0 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("12449")
    inline fun IOType.D0.where(onTrue: Float, onFalse: Float, condition: (IOType.D0) -> IOType.D0) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("29512")
    inline fun IOType.D0.where(condition: IOType.D0, onTrue: Float, onFalse: IOType.D0 = this): IOType.D0 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("4531")
    inline fun IOType.D0.where(onTrue: Float, onFalse: IOType.D0, condition: (IOType.D0) -> IOType.D0) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("26402")
    inline fun IOType.D0.where(condition: IOType.D0, onTrue: IOType.D0 = this, onFalse: Float): IOType.D0 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("15234")
    inline fun IOType.D0.where(onTrue: IOType.D0 = this, onFalse: Float, condition: (IOType.D0) -> IOType.D0) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("4793")
    inline fun IOType.D0.where(condition: IOType.D0, onTrue: IOType.D0 = this, onFalse: IOType.D0 = this): IOType.D0 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("21026")
    inline fun IOType.D0.where(onTrue: IOType.D0 = this, onFalse: IOType.D0, condition: (IOType.D0) -> IOType.D0) =
        where(
            condition = condition(this),
            onTrue = onTrue,
            onFalse = onFalse,
        )

    @JvmName("12453")
    inline fun where(condition: IOType.D1, onTrue: Float, onFalse: Float): IOType.D1 {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("24280")
    inline fun where(condition: IOType.D1, onTrue: Float, onFalse: IOType.D1): IOType.D1 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("3699")
    inline fun where(condition: IOType.D1, onTrue: IOType.D1, onFalse: Float): IOType.D1 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("13977")
    inline fun where(condition: IOType.D1, onTrue: IOType.D1, onFalse: IOType.D1): IOType.D1 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("11855")
    inline fun IOType.D1.where(onTrue: Float, onFalse: Float, condition: (IOType.D1) -> IOType.D1) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("13419")
    inline fun IOType.D1.where(condition: IOType.D1, onTrue: Float, onFalse: IOType.D1 = this): IOType.D1 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("4245")
    inline fun IOType.D1.where(onTrue: Float, onFalse: IOType.D1, condition: (IOType.D1) -> IOType.D1) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("4468")
    inline fun IOType.D1.where(condition: IOType.D1, onTrue: IOType.D1 = this, onFalse: Float): IOType.D1 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("25131")
    inline fun IOType.D1.where(onTrue: IOType.D1 = this, onFalse: Float, condition: (IOType.D1) -> IOType.D1) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("11957")
    inline fun IOType.D1.where(condition: IOType.D1, onTrue: IOType.D1 = this, onFalse: IOType.D1 = this): IOType.D1 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("15666")
    inline fun IOType.D1.where(onTrue: IOType.D1 = this, onFalse: IOType.D1, condition: (IOType.D1) -> IOType.D1) =
        where(
            condition = condition(this),
            onTrue = onTrue,
            onFalse = onFalse,
        )

    @JvmName("26160")
    inline fun where(condition: IOType.D3, onTrue: Float, onFalse: Float): IOType.D3 {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return IOType.D3(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("20892")
    inline fun where(condition: IOType.D3, onTrue: Float, onFalse: IOType.D3): IOType.D3 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D3(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("31423")
    inline fun where(condition: IOType.D3, onTrue: IOType.D3, onFalse: Float): IOType.D3 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D3(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("12220")
    inline fun where(condition: IOType.D3, onTrue: IOType.D3, onFalse: IOType.D3): IOType.D3 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D3(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("19097")
    inline fun IOType.D3.where(onTrue: Float, onFalse: Float, condition: (IOType.D3) -> IOType.D3) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("370")
    inline fun IOType.D3.where(condition: IOType.D3, onTrue: Float, onFalse: IOType.D3 = this): IOType.D3 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("19101")
    inline fun IOType.D3.where(onTrue: Float, onFalse: IOType.D3 = this, condition: (IOType.D3) -> IOType.D3) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("1929")
    inline fun IOType.D3.where(condition: IOType.D3, onTrue: IOType.D3 = this, onFalse: Float): IOType.D3 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("31030")
    inline fun IOType.D3.where(onTrue: IOType.D3 = this, onFalse: Float, condition: (IOType.D3) -> IOType.D3) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("20044")
    inline fun IOType.D3.where(condition: IOType.D3, onTrue: IOType.D3 = this, onFalse: IOType.D3 = this): IOType.D3 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5611")
    inline fun IOType.D3.where(
        onTrue: IOType.D3 = this,
        onFalse: IOType.D3 = this,
        condition: (IOType.D3) -> IOType.D3,
    ) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("12036")
    inline fun IOType.D1.gather(other: IOType.D2): IOType.D2 {
        val result = Backend.gather(x = value, y = other.value, i = 1, j = other.i, k = other.j)
        return IOType.D2(shape = listOf(size, other.j), value = result).also { register(it.value) }
    }

    @JvmName("13835")
    inline fun IOType.D2.scatterAdd(other: IOType.D1, n: Int): IOType.D2 {
        val result = Backend.scatterAdd(x = value, y = other.value, i = 1, j = n, k = j, b = 1)
        return IOType.D2(shape = listOf(n, j), value = result).also { register(it.value) }
    }

    @JvmName("20403")
    inline fun IOType.Companion.d0(value: Float) = IOType.d0(floatArrayOf(value)).also { register(it.value) }

    @JvmName("22432")
    inline fun IOType.Companion.d0(value: FloatArray) = IOType.D0(DataBuffer.create(value)).also { register(it.value) }

    @JvmName("30886")
    inline fun IOType.Companion.d1(size: Int, init: (Int) -> Float = { 0f }): IOType.D1 {
        val value = FloatArray(size)
        for (i in 0 until size) value[i] = init(i)
        return IOType.d1(value = value).also { register(it.value) }
    }

    @JvmName("10727")
    inline fun IOType.Companion.d1(shape: List<Int>, init: (Int) -> Float = { 0f }) =
        d1(shape[0], init).also { register(it.value) }

    @JvmName("11518")
    inline fun IOType.Companion.d1(value: List<Float>) =
        IOType.d1(value = value.toFloatArray()).also { register(it.value) }

    @JvmName("14045")
    inline fun IOType.Companion.d1(vararg elements: Float) = IOType.d1(value = elements).also { register(it.value) }

    @JvmName("29638")
    inline fun IOType.Companion.d1(value: FloatArray) =
        IOType.D1(value = DataBuffer.create(value)).also { register(it.value) }

    @JvmName("16329")
    inline fun IOType.Companion.d2(i: Int, j: Int, init: (Int, Int) -> Float = { _, _ -> 0f }): IOType.D2 {
        val value = FloatArray(i * j)
        for (_i in 0 until i) {
            for (_j in 0 until j) {
                value[_i * j + _j] = init(_i, _j)
            }
        }
        return IOType.d2(shape = listOf(i, j), value = value).also { register(it.value) }
    }

    @JvmName("15314")
    inline fun IOType.Companion.d2(shape: List<Int>, init: (Int, Int) -> Float = { _, _ -> 0f }) = d2(
        i = shape[0],
        j = shape[1],
        init = init,
    ).also { register(it.value) }

    @JvmName("1161")
    inline fun IOType.Companion.d2(shape: List<Int>, value: List<Float>) = IOType.d2(
        value = value.toFloatArray(),
        shape = shape,
    ).also { register(it.value) }

    @JvmName("19369")
    inline fun IOType.Companion.d2(vararg elements: IOType.D1): IOType.D2 {
        val size = elements.size
        val shape = elements.first().shape
        val step = shape.reduce { acc, i -> acc * i }
        val value = DataBuffer.create(size * step)
        elements.forEachIndexed { index, item ->
            val start = index * step
            Backend.copyInto(item.value, value, start until start + item.size)
        }
        return IOType.D2(shape = listOf(size, shape[0]), value = value).also { register(it.value) }
    }

    @JvmName("13688")
    inline fun IOType.Companion.d2(shape: List<Int>, value: FloatArray) =
        IOType.D2(shape = shape, value = DataBuffer.create(value)).also { register(it.value) }

    @JvmName("4025")
    inline fun IOType.Companion.d3(
        i: Int,
        j: Int,
        k: Int,
        init: (Int, Int, Int) -> Float = { _, _, _ -> 0f },
    ): IOType.D3 {
        val value = FloatArray(i * j * k)
        for (_i in 0 until i) {
            for (_j in 0 until j) {
                for (_k in 0 until k) {
                    value[(_i * j + _j) * k + _k] = init(_i, _j, _k)
                }
            }
        }
        return IOType.d3(shape = listOf(i, j, k), value = value).also { register(it.value) }
    }

    @JvmName("652")
    inline fun IOType.Companion.d3(shape: List<Int>, init: (Int, Int, Int) -> Float = { _, _, _ -> 0f }) = d3(
        i = shape[0],
        j = shape[1],
        k = shape[2],
        init = init,
    ).also { register(it.value) }

    @JvmName("10788")
    inline fun IOType.Companion.d3(shape: List<Int>, value: List<Float>) = IOType.d3(
        value = value.toFloatArray(),
        shape = shape,
    ).also { register(it.value) }

    @JvmName("20216")
    inline fun IOType.Companion.d3(vararg elements: IOType.D2): IOType.D3 {
        val size = elements.size
        val shape = elements.first().shape
        val step = shape.reduce { acc, i -> acc * i }
        val value = DataBuffer.create(size * step)
        elements.forEachIndexed { index, item ->
            val start = index * step
            Backend.copyInto(item.value, value, start until start + item.size)
        }
        return IOType.D3(shape = listOf(size, shape[0], shape[1]), value = value).also { register(it.value) }
    }

    @JvmName("8069")
    inline fun IOType.Companion.d3(shape: List<Int>, value: FloatArray) =
        IOType.D3(shape = shape, value = DataBuffer.create(value)).also { register(it.value) }

    @JvmName("20229")
    inline fun IOType.Companion.d4(
        i: Int,
        j: Int,
        k: Int,
        l: Int,
        init: (Int, Int, Int, Int) -> Float = { _, _, _, _ ->
            0f
        },
    ): IOType.D4 {
        val value = FloatArray(i * j * k * l)
        for (_i in 0 until i) {
            for (_j in 0 until j) {
                for (_k in 0 until k) {
                    for (_l in 0 until l) {
                        value[((_i * j + _j) * k + _k) * l + _l] = init(_i, _j, _k, _l)
                    }
                }
            }
        }
        return IOType.d4(shape = listOf(i, j, k, l), value = value).also { register(it.value) }
    }

    @JvmName("13941")
    inline fun IOType.Companion.d4(shape: List<Int>, init: (Int, Int, Int, Int) -> Float = { _, _, _, _ -> 0f }) = d4(
        i = shape[0],
        j = shape[1],
        k = shape[2],
        l = shape[3],
        init = init,
    ).also { register(it.value) }

    @JvmName("32729")
    inline fun IOType.Companion.d4(shape: List<Int>, value: List<Float>) = IOType.d4(
        value = value.toFloatArray(),
        shape = shape,
    ).also { register(it.value) }

    @JvmName("23182")
    inline fun IOType.Companion.d4(vararg elements: IOType.D3): IOType.D4 {
        val size = elements.size
        val shape = elements.first().shape
        val step = shape.reduce { acc, i -> acc * i }
        val value = DataBuffer.create(size * step)
        elements.forEachIndexed { index, item ->
            val start = index * step
            Backend.copyInto(item.value, value, start until start + item.size)
        }
        return IOType.D4(shape = listOf(size, shape[0], shape[1], shape[2]), value = value).also { register(it.value) }
    }

    @JvmName("13800")
    inline fun IOType.Companion.d4(shape: List<Int>, value: FloatArray) =
        IOType.D4(shape = shape, value = DataBuffer.create(value)).also { register(it.value) }

    /**
     * get
     */
    @JvmName("10700")
    inline fun IOType.D0.get() = value[0]

    @JvmName("15021")
    inline operator fun IOType.D1.get(index: Int) = value[index]

    @JvmName("28821")
    inline operator fun IOType.D2.get(i: Int, j: Int) = value[i * shape[1] + j]

    @JvmName("12506")
    inline operator fun IOType.D2.get(i: Int): IOType.D1 {
        val offset = i * shape[1]
        val result = Backend.slice(x = value, indices = offset until offset + shape[1])
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("20271")
    inline operator fun IOType.D3.get(i: Int, j: Int, k: Int) = value[(i * shape[1] + j) * shape[2] + k]

    @JvmName("13392")
    inline operator fun IOType.D3.get(i: Int, j: Int): IOType.D1 {
        val offset = (i * shape[1] + j) * shape[2]
        val result = Backend.slice(x = value, indices = offset until offset + shape[2])
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("25176")
    inline operator fun IOType.D3.get(i: Int): IOType.D2 {
        val offset = i * shape[1] * shape[2]
        val result = Backend.slice(x = value, indices = offset until offset + shape[1] * shape[2])
        return IOType.D2(
            shape = listOf(shape[1], shape[2]),
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("29980")
    inline operator fun IOType.D4.get(i: Int, j: Int, k: Int, l: Int) =
        value[((i * shape[1] + j) * shape[2] + k) * shape[3] + l]

    @JvmName("10453")
    inline operator fun IOType.D4.get(i: Int, j: Int, k: Int): IOType.D1 {
        val offset = ((i * shape[1] + j) * shape[2] + k) * shape[3]
        val result = Backend.slice(x = value, indices = offset until offset + shape[3])
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("6107")
    inline operator fun IOType.D4.get(i: Int, j: Int): IOType.D2 {
        val offset = (i * shape[1] + j) * shape[2] * shape[3]
        val result = Backend.slice(x = value, indices = offset until offset + shape[2] * shape[3])
        return IOType.D2(
            shape = listOf(shape[2], shape[3]),
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("30111")
    inline operator fun IOType.D4.get(i: Int): IOType.D3 {
        val offset = i * shape[1] * shape[2] * shape[3]
        val result = Backend.slice(x = value, indices = offset until offset + shape[1] * shape[2] * shape[3])
        return IOType.D3(
            shape = listOf(shape[1], shape[2], shape[3]),
            value = result,
        ).also { register(it.value) }
    }

    /**
     * set
     */
    @JvmName("25833")
    inline fun IOType.D0.set(element: Float) {
        value[0] = element
    }

    @JvmName("14093")
    inline operator fun IOType.D1.set(index: Int, element: Float) {
        value[index] = element
    }

    @JvmName("22118")
    inline operator fun IOType.D2.set(i: Int, j: Int, element: Float) {
        value[i * shape[1] + j] = element
    }

    @JvmName("5402")
    inline operator fun IOType.D2.set(i: Int, element: IOType.D1) {
        val start = i * shape[1]
        Backend.copyInto(
            x = element.value,
            y = value,
            indices = start until start + element.value.size,
        )
    }

    @JvmName("20122")
    inline operator fun IOType.D3.set(i: Int, j: Int, z: Int, element: Float) {
        value[(i * shape[1] + j) * shape[2] + z] = element
    }

    @JvmName("3948")
    inline operator fun IOType.D3.set(i: Int, j: Int, element: IOType.D1) {
        val start = (i * shape[1] + j) * shape[2]
        Backend.copyInto(
            x = element.value,
            y = value,
            indices = start until start + element.value.size,
        )
    }

    @JvmName("31824")
    inline operator fun IOType.D3.set(i: Int, element: IOType.D2) {
        val start = i * shape[1] * shape[2]
        Backend.copyInto(
            x = element.value,
            y = value,
            indices = start until start + element.value.size,
        )
    }

    @JvmName("16626")
    inline operator fun IOType.D4.set(i: Int, j: Int, k: Int, l: Int, element: Float) {
        value[((i * shape[1] + j) * shape[2] + k) * shape[3] + l] = element
    }

    @JvmName("11796")
    inline fun Batch<IOType.D1>.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))

    @JvmName("1298")
    inline fun Batch<IOType.D1>.reshapeToD2(shape: List<Int>) =
        Batch<IOType.D2>(size = size, shape = shape, value = value)

    @JvmName("3399")
    inline fun Batch<IOType.D1>.reshapeToD3(i: Int, j: Int, k: Int) = reshapeToD3(listOf(i, j, k))

    @JvmName("20140")
    inline fun Batch<IOType.D1>.reshapeToD3(shape: List<Int>) =
        Batch<IOType.D3>(size = size, shape = shape, value = value)

    @JvmName("8977")
    inline fun Batch<IOType.D2>.reshapeToD3(i: Int, j: Int, k: Int) = reshapeToD3(listOf(i, j, k))

    @JvmName("29664")
    inline fun Batch<IOType.D2>.reshapeToD3(shape: List<Int>) =
        Batch<IOType.D3>(size = size, shape = shape, value = value)

    @JvmName("27839")
    inline fun Batch<IOType.D3>.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))

    @JvmName("16255")
    inline fun Batch<IOType.D3>.reshapeToD2(shape: List<Int>) =
        Batch<IOType.D2>(size = size, shape = shape, value = value)

    @JvmName("792")
    inline fun Batch<IOType.D1>.slice(indices: IntProgression): Batch<IOType.D1> {
        val result = Backend.slice(x = value, xi = size, xj = shape[0], axis = 1, indices = indices)
        return Batch<IOType.D1>(size = size, shape = listOf(indices.size), value = result).also { register(it.value) }
    }

    @JvmName("10170")
    inline fun Batch<IOType.D2>.slice(indices: IntProgression, axis: Int): Batch<IOType.D2> {
        val result =
            Backend.slice(x = value, xi = size, xj = shape[0], xk = shape[1], axis = axis + 1, indices = indices)
        return Batch<IOType.D2>(
            size = size,
            shape = when (axis) {
                0 -> listOf(indices.size, shape[1])
                else -> listOf(shape[0], indices.size)
            },
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("32341")
    inline fun Batch<IOType.D2>.transpose(): Batch<IOType.D2> {
        val result =
            Backend.transpose(x = value, xi = size, xj = shape[0], xk = shape[1], axisI = 0, axisJ = 2, axisK = 1)
        return Batch<IOType.D2>(size = size, shape = shape.reversed(), value = result).also { register(it.value) }
    }

    @JvmName("19007")
    inline fun Batch<IOType.D3>.transpose(axisI: Int, axisJ: Int, axisK: Int): Batch<IOType.D3> {
        val result = Backend.transpose(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            axisI = 0,
            axisJ = axisI + 1,
            axisK = axisJ + 1,
            axisL = axisK + 1,
        )
        return Batch<IOType.D3>(
            size = size,
            shape = listOf(shape[axisI], shape[axisJ], shape[axisK]),
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("28290")
    inline fun Batch<IOType.D2>.flatten() = Batch<IOType.D1>(
        shape = listOf(step),
        size = size,
        value = value,
    )

    @JvmName("19562")
    inline fun Batch<IOType.D3>.flatten() = Batch<IOType.D1>(
        shape = listOf(step),
        size = size,
        value = value,
    )

    /**
     * Unfold: Batch<IOType.D2>を列形式に展開 (im2col)
     * 入力: [batchSize] x [channel, inputSize]
     * 出力: [windowSize * channel, outputSize * batchSize]
     */

    @JvmName("30701")
    inline fun Batch<IOType.D2>.unfold(windowSize: Int, stride: Int, padding: Int): IOType.D2 {
        val channel = shape[0]
        val inputSize = shape[1]
        val outputSize = (inputSize - windowSize + 2 * padding) / stride + 1
        val row = windowSize * channel
        val column = outputSize * size
        val result = IOType.d2(row, column)

        for (batchIndex in 0 until size) {
            val input = this[batchIndex]
            for (rowIdx in 0 until row) {
                val channelIndex = rowIdx / windowSize
                val windowIndex = rowIdx % windowSize
                for (colIdx in 0 until outputSize) {
                    val columnIndex = batchIndex * outputSize + colIdx
                    val inputIdx = colIdx * stride + windowIndex - padding
                    if (inputIdx in 0 until inputSize) {
                        result[rowIdx, columnIndex] = input[channelIndex, inputIdx]
                    }
                }
            }
        }
        return result.also { register(it.value) }
    }

    /**
     * Fold: 列形式をBatch<IOType.D2>に戻す (col2im)
     * 入力: [windowSize * channel, outputSize * batchSize]
     * 出力: [batchSize] x [channel, inputSize]
     * 注意: 重複部分は加算される
     */
    @JvmName("24551")
    inline fun IOType.D2.fold(
        batchSize: Int,
        channel: Int,
        inputSize: Int,
        stride: Int,
        padding: Int,
    ): Batch<IOType.D2> {
        val windowSize = shape[0] / channel
        val outputSize = shape[1] / batchSize
        return Batch(batchSize) { b ->
            IOType.d2(channel, inputSize) { c, i ->
                var sum = 0f
                for (outputIdx in 0 until outputSize) {
                    val windowIndex = i - outputIdx * stride + padding
                    if (windowIndex in 0 until windowSize) {
                        val row = c * windowSize + windowIndex
                        val col = b * outputSize + outputIdx
                        sum += this[row, col]
                    }
                }
                sum
            }
        }
    }

    /**
     * Unfold: Batch<IOType.D3>を列形式に展開 (im2col)
     * 入力: [batchSize] x [channel, inputSizeX, inputSizeY]
     * 出力: [windowSize * windowSize * channel, outputSizeX * outputSizeY * batchSize]
     */

    @JvmName("16224")
    inline fun Batch<IOType.D3>.unfold(windowSize: Int, stride: Int, padding: Int): IOType.D2 {
        val channel = shape[0]
        val inputX = shape[1]
        val inputY = shape[2]
        val outputX = (inputX - windowSize + 2 * padding) / stride + 1
        val outputY = (inputY - windowSize + 2 * padding) / stride + 1
        val row = windowSize * windowSize * channel
        val column = outputX * outputY * size
        val result = IOType.d2(row, column)

        for (batchIndex in 0 until size) {
            val input = this[batchIndex]
            for (c in 0 until channel) {
                for (wy in 0 until windowSize) {
                    for (wx in 0 until windowSize) {
                        val rowIdx = c * windowSize * windowSize + wy * windowSize + wx
                        for (oy in 0 until outputY) {
                            for (ox in 0 until outputX) {
                                val columnIndex = batchIndex * outputX * outputY + oy * outputX + ox
                                val inputIdxX = ox * stride + wx - padding
                                val inputIdxY = oy * stride + wy - padding
                                if (inputIdxX in 0 until inputX && inputIdxY in 0 until inputY) {
                                    result[rowIdx, columnIndex] = input[c, inputIdxX, inputIdxY]
                                }
                            }
                        }
                    }
                }
            }
        }
        return result.also { register(it.value) }
    }

    /**
     * Fold: 列形式をBatch<IOType.D3>に戻す (col2im)
     * 入力: [windowSize * windowSize * channel, outputSizeX * outputSizeY * batchSize]
     * 出力: [batchSize] x [channel, inputSizeX, inputSizeY]
     * 注意: 重複部分は加算される
     */
    @JvmName("4694")
    inline fun IOType.D2.fold(
        batchSize: Int,
        channel: Int,
        inputX: Int,
        inputY: Int,
        stride: Int,
        padding: Int,
    ): Batch<IOType.D3> {
        val windowSize = kotlin.math.sqrt((shape[0] / channel).toDouble()).toInt()
        val outputSizeXY = shape[1] / batchSize
        val outputSizeX = kotlin.math.sqrt(outputSizeXY.toDouble()).toInt()
        val outputSizeY = outputSizeXY / outputSizeX

        return Batch(batchSize) { b ->
            IOType.d3(channel, inputX, inputY) { c, ix, iy ->
                var sum = 0f
                for (oy in 0 until outputSizeY) {
                    for (ox in 0 until outputSizeX) {
                        val windowIdxX = ix - ox * stride + padding
                        val windowIdxY = iy - oy * stride + padding
                        if (windowIdxX in 0 until windowSize && windowIdxY in 0 until windowSize) {
                            val row = c * windowSize * windowSize + windowIdxY * windowSize + windowIdxX
                            val col = b * outputSizeX * outputSizeY + oy * outputSizeX + ox
                            sum += this[row, col]
                        }
                    }
                }
                sum
            }
        }
    }

    @JvmName("4231")
    inline fun Batch<IOType.D1>.broadcastToD2(axis: Int, size: Int) =
        Batch(this.size) { this[it].broadcastToD2(axis, size).also { register(it.value) } }

    @JvmName("3899")
    inline fun Batch<IOType.D2>.broadcastToD3(axis: Int, size: Int) =
        Batch(this.size) { this[it].broadcastToD3(axis, size).also { register(it.value) } }

    @JvmName("15031")
    inline fun Batch<IOType.D1>.interleave(other: Batch<IOType.D1>): Batch<IOType.D1> {
        check(size == other.size && shape[0] == other.shape[0])
        val i = shape[0] * 2
        val result = DataBuffer.create(value.size + other.value.size)
        Backend.copyInto(
            x = value,
            y = result,
            yi = size,
            yj = i,
            axis = 1,
            indices = 0 until i step 2,
        )
        Backend.copyInto(
            x = other.value,
            y = result,
            yi = size,
            yj = i,
            axis = 1,
            indices = 1 until i step 2,
        )
        return Batch<IOType.D1>(size = size, shape = listOf(i), value = result).also { register(it.value) }
    }

    @JvmName("29022")
    inline fun Batch<IOType.D2>.interleave(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
        check(size == other.size && shape[0] == other.shape[0] && shape[1] == other.shape[1])
        val result = DataBuffer.create(value.size + other.value.size)
        val newShape = when (axis) {
            0 -> listOf(shape[0] * 2, shape[1])
            else -> listOf(shape[0], shape[1] * 2)
        }
        val (i, j) = newShape
        Backend.copyInto(
            x = value,
            y = result,
            yi = size,
            yj = i,
            yk = j,
            axis = axis + 1,
            indices = when (axis) {
                0 -> 0 until i step 2
                else -> 0 until j step 2
            },
        )
        Backend.copyInto(
            x = other.value,
            y = result,
            yi = size,
            yj = i,
            yk = j,
            axis = axis + 1,
            indices = when (axis) {
                0 -> 1 until i step 2
                else -> 1 until j step 2
            },
        )
        return Batch<IOType.D2>(
            size = size,
            shape = newShape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("3668")
    inline fun Batch<IOType.D1>.toD2(): IOType.D2 =
        IOType.D2(shape = listOf(size, shape[0]), value = value)

    @JvmName("21932")
    inline fun IOType.D2.toD1(): Batch<IOType.D1> =
        Batch<IOType.D1>(value = value, size = shape[0], shape = listOf(shape[1]))

    @JvmName("21951")
    inline fun Batch<IOType.D2>.toD3(): IOType.D3 =
        IOType.D3(shape = listOf(size, shape[0], shape[1]), value = value)

    @JvmName("23111")
    inline fun IOType.D3.toBatch(): Batch<IOType.D2> =
        Batch<IOType.D2>(value = value, size = shape[0], shape = listOf(shape[1], shape[2]))

    @JvmName("28718")
    inline fun Batch<IOType.D3>.toD4(): IOType.D4 =
        IOType.D4(shape = listOf(size, shape[0], shape[1], shape[2]), value = value)

    @JvmName("2534")
    inline fun IOType.D4.toBatch(): Batch<IOType.D3> =
        Batch(value = value, size = shape[0], shape = listOf(shape[1], shape[2], shape[3]))

    @JvmName("18885")
    inline operator fun Batch<IOType.D2>.div(other: Float): Batch<IOType.D2> {
        val result = Backend.div(x = value, y = other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("29651")
    inline operator fun Batch<IOType.D2>.div(other: Batch<IOType.D0>): Batch<IOType.D2> {
        val result = Backend.div(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("4205")
    inline fun Batch<IOType.D2>.div(other: IOType.D1, axis: Int): Batch<IOType.D2> {
        val result = Backend.div(x = value, xi = size, xj = shape[0], xk = shape[1], y = other.value, axis = axis + 1)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("22296")
    inline fun Batch<IOType.D2>.div(other: Batch<IOType.D1>, axis: Int): Batch<IOType.D2> {
        val result = Backend.div(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            axis1 = 0,
            axis2 = axis + 1,
        )
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("2845")
    inline operator fun Batch<IOType.D2>.div(other: IOType.D2): Batch<IOType.D2> {
        val result = Backend.div(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 1,
        )
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("27653")
    inline operator fun Batch<IOType.D2>.div(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.div(x = value, y = other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("23914")
    inline fun Batch<IOType.D2>.div(other: Batch<IOType.D3>, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.div(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            yl = other.shape[2],
            axis1 = 0,
            axis2 = axis1 + 1,
            axis3 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("19728")
    inline operator fun Float.div(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.div(x = this, y = other.value)
        return Batch<IOType.D0>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("31729")
    inline operator fun Float.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.div(x = this, y = other.value)
        return Batch<IOType.D1>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("29698")
    inline operator fun Float.div(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.div(x = this, y = other.value)
        return Batch<IOType.D2>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("13720")
    inline operator fun Float.div(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.div(x = this, y = other.value)
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("10571")
    inline operator fun Batch<IOType.D0>.div(other: Float): Batch<IOType.D0> {
        val result = Backend.div(x = value, y = other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("21317")
    inline operator fun Batch<IOType.D0>.div(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.div(x = value, y = other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("17998")
    inline operator fun Batch<IOType.D0>.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.div(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
        return Batch<IOType.D1>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("1842")
    inline operator fun Batch<IOType.D0>.div(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.div(
            x = value,
            y = other.value,
            yi = size,
            yj = other.step,
            axis = 0,
        )
        return Batch<IOType.D2>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("4827")
    inline operator fun Batch<IOType.D0>.div(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.div(
            x = value,
            y = other.value,
            yi = size,
            yj = other.step,
            axis = 0,
        )
        return Batch<IOType.D3>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("8890")
    inline operator fun Batch<IOType.D1>.div(other: Float): Batch<IOType.D1> {
        val result = Backend.div(x = value, y = other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("11822")
    inline operator fun Batch<IOType.D1>.div(other: Batch<IOType.D0>): Batch<IOType.D1> {
        val result = Backend.div(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 0,
        )
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("15520")
    inline operator fun Batch<IOType.D1>.div(other: IOType.D1): Batch<IOType.D1> {
        val result = Backend.div(x = value, xi = size, xj = step, y = other.value, axis = 1)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("15913")
    inline operator fun Batch<IOType.D1>.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.div(x = value, y = other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("15823")
    inline fun Batch<IOType.D1>.div(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
        val result = Backend.div(
            x = value,
            xi = size,
            xj = shape[0],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            axis1 = 0,
            axis2 = axis + 1,
        )
        return Batch<IOType.D2>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("9093")
    inline operator fun Batch<IOType.D3>.div(other: Float): Batch<IOType.D3> {
        val result = Backend.div(x = value, y = other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5809")
    inline operator fun Batch<IOType.D3>.div(other: Batch<IOType.D0>): Batch<IOType.D3> {
        val result = Backend.div(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("933")
    inline fun Batch<IOType.D3>.div(other: IOType.D1, axis: Int): Batch<IOType.D3> {
        val result = Backend.div(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            axis = axis + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("31929")
    inline operator fun Batch<IOType.D3>.div(other: IOType.D2): Batch<IOType.D3> =
        div(other = other, axis1 = 1, axis2 = 2)

    @JvmName("4148")
    inline fun Batch<IOType.D3>.div(other: IOType.D2, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.div(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            yi = other.shape[0],
            yj = other.shape[1],
            axis1 = axis1 + 1,
            axis2 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("10517")
    inline operator fun Batch<IOType.D3>.div(other: Batch<IOType.D2>) = div(other, axis1 = 1, axis2 = 2)

    @JvmName("26064")
    inline fun Batch<IOType.D3>.div(other: Batch<IOType.D2>, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.div(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            axis1 = 0,
            axis2 = axis1 + 1,
            axis3 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("12043")
    inline operator fun Batch<IOType.D3>.div(other: IOType.D3): Batch<IOType.D3> {
        val result = Backend.div(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3323")
    inline operator fun Batch<IOType.D3>.div(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.div(x = value, y = other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("17753")
    inline operator fun Batch<IOType.D2>.minus(other: Float): Batch<IOType.D2> {
        val result = Backend.minus(x = value, y = other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("23301")
    inline operator fun Batch<IOType.D2>.minus(other: Batch<IOType.D0>): Batch<IOType.D2> {
        val result = Backend.minus(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("10173")
    inline fun Batch<IOType.D2>.minus(other: IOType.D1, axis: Int): Batch<IOType.D2> {
        val result = Backend.minus(x = value, xi = size, xj = shape[0], xk = shape[1], y = other.value, axis = axis + 1)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3196")
    inline fun Batch<IOType.D2>.minus(other: Batch<IOType.D1>, axis: Int): Batch<IOType.D2> {
        val result = Backend.minus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            axis1 = 0,
            axis2 = axis + 1,
        )
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("27156")
    inline operator fun Batch<IOType.D2>.minus(other: IOType.D2): Batch<IOType.D2> {
        val result = Backend.minus(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 1,
        )
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("9043")
    inline operator fun Batch<IOType.D2>.minus(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.minus(x = value, y = other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("27460")
    inline fun Batch<IOType.D2>.minus(other: Batch<IOType.D3>, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.minus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            yl = other.shape[2],
            axis1 = 0,
            axis2 = axis1 + 1,
            axis3 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("20035")
    inline operator fun Float.minus(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.minus(x = this, y = other.value)
        return Batch<IOType.D0>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("1824")
    inline operator fun Float.minus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.minus(x = this, y = other.value)
        return Batch<IOType.D1>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("9879")
    inline operator fun Float.minus(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.minus(x = this, y = other.value)
        return Batch<IOType.D2>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("22375")
    inline operator fun Float.minus(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.minus(x = this, y = other.value)
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("5242")
    inline operator fun Batch<IOType.D0>.minus(other: Float): Batch<IOType.D0> {
        val result = Backend.minus(x = value, y = other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("25030")
    inline operator fun Batch<IOType.D0>.minus(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.minus(x = value, y = other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("10044")
    inline operator fun Batch<IOType.D0>.minus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.minus(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
        return Batch<IOType.D1>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("26430")
    inline operator fun Batch<IOType.D0>.minus(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.minus(
            x = value,
            y = other.value,
            yi = size,
            yj = other.step,
            axis = 0,
        )
        return Batch<IOType.D2>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("5042")
    inline operator fun Batch<IOType.D0>.minus(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.minus(
            x = value,
            y = other.value,
            yi = size,
            yj = other.step,
            axis = 0,
        )
        return Batch<IOType.D3>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("20676")
    inline operator fun Batch<IOType.D1>.minus(other: Float): Batch<IOType.D1> {
        val result = Backend.minus(x = value, y = other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("28813")
    inline operator fun Batch<IOType.D1>.minus(other: Batch<IOType.D0>): Batch<IOType.D1> {
        val result = Backend.minus(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 0,
        )
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13957")
    inline operator fun Batch<IOType.D1>.minus(other: IOType.D1): Batch<IOType.D1> {
        val result = Backend.minus(x = value, xi = size, xj = step, y = other.value, axis = 1)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("16288")
    inline operator fun Batch<IOType.D1>.minus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.minus(x = value, y = other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("15902")
    inline fun Batch<IOType.D1>.minus(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
        val result = Backend.minus(
            x = value,
            xi = size,
            xj = shape[0],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            axis1 = 0,
            axis2 = axis + 1,
        )
        return Batch<IOType.D2>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("20914")
    inline operator fun Batch<IOType.D3>.minus(other: Float): Batch<IOType.D3> {
        val result = Backend.minus(x = value, y = other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("23704")
    inline operator fun Batch<IOType.D3>.minus(other: Batch<IOType.D0>): Batch<IOType.D3> {
        val result = Backend.minus(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("24776")
    inline fun Batch<IOType.D3>.minus(other: IOType.D1, axis: Int): Batch<IOType.D3> {
        val result = Backend.minus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            axis = axis + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("20930")
    inline operator fun Batch<IOType.D3>.minus(other: IOType.D2): Batch<IOType.D3> =
        minus(other = other, axis1 = 1, axis2 = 2)

    @JvmName("17820")
    inline fun Batch<IOType.D3>.minus(other: IOType.D2, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.minus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            yi = other.shape[0],
            yj = other.shape[1],
            axis1 = axis1 + 1,
            axis2 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("22013")
    inline operator fun Batch<IOType.D3>.minus(other: Batch<IOType.D2>) = minus(other, axis1 = 1, axis2 = 2)

    @JvmName("26330")
    inline fun Batch<IOType.D3>.minus(other: Batch<IOType.D2>, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.minus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            axis1 = 0,
            axis2 = axis1 + 1,
            axis3 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("28239")
    inline operator fun Batch<IOType.D3>.minus(other: IOType.D3): Batch<IOType.D3> {
        val result = Backend.minus(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13312")
    inline operator fun Batch<IOType.D3>.minus(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.minus(x = value, y = other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5850")
    inline infix fun Batch<IOType.D1>.inner(other: Batch<IOType.D1>): Batch<IOType.D0> {
        val result = Backend.inner(x = value, y = other.value, b = size)
        return Batch<IOType.D0>(value = result, size = size, shape = listOf(1)).also { register(it.value) }
    }

    @JvmName("20231")
    inline operator fun IOType.D2.times(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.times(
            x = value,
            y = other.value,
            yi = other.size,
            yj = other.step,
            axis = 1,
        )
        return Batch<IOType.D2>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("32398")
    inline operator fun Batch<IOType.D2>.times(other: Float): Batch<IOType.D2> {
        val result = Backend.times(x = value, y = other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("10046")
    inline operator fun Batch<IOType.D2>.times(other: Batch<IOType.D0>): Batch<IOType.D2> {
        val result = Backend.times(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("15520")
    inline fun Batch<IOType.D2>.times(other: IOType.D1, axis: Int): Batch<IOType.D2> {
        val result = Backend.times(x = value, xi = size, xj = shape[0], xk = shape[1], y = other.value, axis = axis + 1)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("8072")
    inline fun Batch<IOType.D2>.times(other: Batch<IOType.D1>, axis: Int): Batch<IOType.D2> {
        val result = Backend.times(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            axis1 = 0,
            axis2 = axis + 1,
        )
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("28966")
    inline operator fun Batch<IOType.D2>.times(other: IOType.D2): Batch<IOType.D2> {
        val result = Backend.times(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 1,
        )
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("18535")
    inline operator fun Batch<IOType.D2>.times(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.times(x = value, y = other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("26381")
    inline fun Batch<IOType.D2>.times(other: Batch<IOType.D3>, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.times(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            yl = other.shape[2],
            axis1 = 0,
            axis2 = axis1 + 1,
            axis3 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("11785")
    inline operator fun Float.times(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.times(x = this, y = other.value)
        return Batch<IOType.D0>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("21721")
    inline operator fun Float.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.times(x = this, y = other.value)
        return Batch<IOType.D1>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("22584")
    inline operator fun Float.times(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.times(x = this, y = other.value)
        return Batch<IOType.D2>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("32351")
    inline operator fun Float.times(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.times(x = this, y = other.value)
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("1573")
    inline operator fun Batch<IOType.D0>.times(other: Float): Batch<IOType.D0> {
        val result = Backend.times(x = value, y = other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("7159")
    inline operator fun Batch<IOType.D0>.times(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.times(x = value, y = other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("2593")
    inline operator fun Batch<IOType.D0>.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.times(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
        return Batch<IOType.D1>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("26404")
    inline operator fun Batch<IOType.D0>.times(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.times(
            x = value,
            y = other.value,
            yi = size,
            yj = other.step,
            axis = 0,
        )
        return Batch<IOType.D2>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("14109")
    inline operator fun Batch<IOType.D0>.times(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.times(
            x = value,
            y = other.value,
            yi = size,
            yj = other.step,
            axis = 0,
        )
        return Batch<IOType.D3>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("27316")
    inline operator fun IOType.D1.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.times(
            x = value,
            y = other.value,
            yi = other.size,
            yj = other.step,
            axis = 1,
        )
        return Batch<IOType.D1>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("8489")
    inline operator fun Batch<IOType.D1>.times(other: Float): Batch<IOType.D1> {
        val result = Backend.times(x = value, y = other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("11645")
    inline operator fun Batch<IOType.D1>.times(other: Batch<IOType.D0>): Batch<IOType.D1> {
        val result = Backend.times(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 0,
        )
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("24909")
    inline operator fun Batch<IOType.D1>.times(other: IOType.D1): Batch<IOType.D1> {
        val result = Backend.times(x = value, xi = size, xj = step, y = other.value, axis = 1)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("19682")
    inline operator fun Batch<IOType.D1>.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.times(x = value, y = other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("12338")
    inline fun Batch<IOType.D1>.times(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
        val result = Backend.times(
            x = value,
            xi = size,
            xj = shape[0],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            axis1 = 0,
            axis2 = axis + 1,
        )
        return Batch<IOType.D2>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("5981")
    inline operator fun IOType.D3.times(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.times(
            x = value,
            y = other.value,
            yi = other.size,
            yj = other.step,
            axis = 1,
        )
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("31076")
    inline operator fun Batch<IOType.D3>.times(other: Float): Batch<IOType.D3> {
        val result = Backend.times(x = value, y = other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("9378")
    inline operator fun Batch<IOType.D3>.times(other: Batch<IOType.D0>): Batch<IOType.D3> {
        val result = Backend.times(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("26781")
    inline fun Batch<IOType.D3>.times(other: IOType.D1, axis: Int): Batch<IOType.D3> {
        val result = Backend.times(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            axis = axis + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("20536")
    inline operator fun Batch<IOType.D3>.times(other: IOType.D2): Batch<IOType.D3> =
        times(other = other, axis1 = 1, axis2 = 2)

    @JvmName("28882")
    inline fun Batch<IOType.D3>.times(other: IOType.D2, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.times(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            yi = other.shape[0],
            yj = other.shape[1],
            axis1 = axis1 + 1,
            axis2 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13985")
    inline operator fun Batch<IOType.D3>.times(other: Batch<IOType.D2>) = times(other, axis1 = 1, axis2 = 2)

    @JvmName("9892")
    inline fun Batch<IOType.D3>.times(other: Batch<IOType.D2>, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.times(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            axis1 = 0,
            axis2 = axis1 + 1,
            axis3 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("7531")
    inline operator fun Batch<IOType.D3>.times(other: IOType.D3): Batch<IOType.D3> {
        val result = Backend.times(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("20106")
    inline operator fun Batch<IOType.D3>.times(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.times(x = value, y = other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("7039")
    inline fun IOType.D2.matMul(other: Batch<IOType.D1>, trans: Boolean = false): Batch<IOType.D1> {
        val n = if (trans) shape[1] else shape[0]
        val k = if (trans) shape[0] else shape[1]
        val result = Backend.matMul(
            x = other.value,
            transX = false,
            y = value,
            transY = !trans,
            m = other.size,
            n = n,
            k = k,
            b = 1,
        )

        return Batch<IOType.D1>(result, other.size, listOf(n)).also { register(it.value) }
    }

    @JvmName("30540")
    inline fun Batch<IOType.D2>.matMul(
        other: IOType.D2,
        transA: Boolean = false,
        transB: Boolean = false,
    ): Batch<IOType.D2> {
        val m = if (transA) shape[1] else shape[0]
        val n = if (transB) other.shape[0] else other.shape[1]
        val k = if (transA) shape[0] else shape[1]
        val result = Backend.matMul(
            x = value,
            transX = transA,
            y = other.value,
            transY = transB,
            m = size * m,
            n = n,
            k = k,
            b = 1,
        )
        return Batch<IOType.D2>(value = result, size = size, shape = listOf(m, n)).also { register(it.value) }
    }

    @JvmName("8442")
    inline fun Batch<IOType.D2>.matMul(
        other: Batch<IOType.D2>,
        transA: Boolean = false,
        transB: Boolean = false,
    ): Batch<IOType.D2> {
        val m = if (transA) shape[1] else shape[0]
        val n = if (transB) other.shape[0] else other.shape[1]
        val k = if (transA) shape[0] else shape[1]
        val result = Backend.matMul(
            x = value,
            transX = transA,
            y = other.value,
            transY = transB,
            m = m,
            n = n,
            k = k,
            b = size,
        )
        return Batch<IOType.D2>(value = result, size = size, shape = listOf(m, n)).also { register(it.value) }
    }

    @JvmName("14291")
    inline fun IOType.D3.matMul(
        other: Batch<IOType.D3>,
        transA: Boolean = false,
        transB: Boolean = false,
    ): Batch<IOType.D3> {
        val m = if (transA) shape[2] else shape[1]
        val n = if (transB) other.shape[1] else other.shape[2]
        val k = if (transA) shape[1] else shape[2]
        val result = Backend.matMul(
            x = value,
            transX = transA,
            y = other.value,
            transY = transB,
            m = m,
            n = size * n,
            k = k,
            b = shape[0],
        )
        return Batch<IOType.D3>(value = result, size = size, shape = listOf(shape[0], m, n)).also { register(it.value) }
    }

    @JvmName("18514")
    inline fun Batch<IOType.D3>.matMul(
        other: IOType.D3,
        transA: Boolean = false,
        transB: Boolean = false,
    ): Batch<IOType.D3> {
        val m = if (transA) shape[2] else shape[1]
        val n = if (transB) other.shape[1] else other.shape[2]
        val k = if (transA) shape[1] else shape[2]
        val result = Backend.matMul(
            x = value,
            transX = transA,
            y = other.value,
            transY = transB,
            m = size * m,
            n = n,
            k = k,
            b = shape[0],
        )
        return Batch<IOType.D3>(value = result, size = size, shape = listOf(shape[0], m, n)).also { register(it.value) }
    }

    @JvmName("23200")
    inline fun Batch<IOType.D3>.matMul(
        other: Batch<IOType.D3>,
        transA: Boolean = false,
        transB: Boolean = false,
    ): Batch<IOType.D3> {
        val m = if (transA) shape[2] else shape[1]
        val n = if (transB) other.shape[1] else other.shape[2]
        val k = if (transA) shape[1] else shape[2]
        val result = Backend.matMul(
            x = value,
            transX = transA,
            y = other.value,
            transY = transB,
            m = m,
            n = n,
            k = k,
            b = size * shape[0],
        )
        return Batch<IOType.D3>(value = result, size = size, shape = listOf(shape[0], m, n)).also { register(it.value) }
    }

    @JvmName("14067")
    inline operator fun Batch<IOType.D2>.plus(other: Float): Batch<IOType.D2> {
        val result = Backend.plus(x = value, y = other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("15219")
    inline operator fun Batch<IOType.D2>.plus(other: Batch<IOType.D0>): Batch<IOType.D2> {
        val result = Backend.plus(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("27523")
    inline fun Batch<IOType.D2>.plus(other: IOType.D1, axis: Int): Batch<IOType.D2> {
        val result = Backend.plus(x = value, xi = size, xj = shape[0], xk = shape[1], y = other.value, axis = axis + 1)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5862")
    inline fun Batch<IOType.D2>.plus(other: Batch<IOType.D1>, axis: Int): Batch<IOType.D2> {
        val result = Backend.plus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            axis1 = 0,
            axis2 = axis + 1,
        )
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("10652")
    inline operator fun Batch<IOType.D2>.plus(other: IOType.D2): Batch<IOType.D2> {
        val result = Backend.plus(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 1,
        )
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("17777")
    inline operator fun Batch<IOType.D2>.plus(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.plus(x = value, y = other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("17542")
    inline fun Batch<IOType.D2>.plus(other: Batch<IOType.D3>, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.plus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            yl = other.shape[2],
            axis1 = 0,
            axis2 = axis1 + 1,
            axis3 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("7609")
    inline operator fun Float.plus(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.plus(x = this, y = other.value)
        return Batch<IOType.D0>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("6981")
    inline operator fun Float.plus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.plus(x = this, y = other.value)
        return Batch<IOType.D1>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("31436")
    inline operator fun Float.plus(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.plus(x = this, y = other.value)
        return Batch<IOType.D2>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("30300")
    inline operator fun Float.plus(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.plus(x = this, y = other.value)
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("379")
    inline operator fun Batch<IOType.D0>.plus(other: Float): Batch<IOType.D0> {
        val result = Backend.plus(x = value, y = other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("2801")
    inline operator fun Batch<IOType.D0>.plus(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.plus(x = value, y = other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("24885")
    inline operator fun Batch<IOType.D0>.plus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.plus(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
        return Batch<IOType.D1>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("8405")
    inline operator fun Batch<IOType.D0>.plus(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.plus(
            x = value,
            y = other.value,
            yi = size,
            yj = other.step,
            axis = 0,
        )
        return Batch<IOType.D2>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("19351")
    inline operator fun Batch<IOType.D0>.plus(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.plus(
            x = value,
            y = other.value,
            yi = size,
            yj = other.step,
            axis = 0,
        )
        return Batch<IOType.D3>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("32765")
    inline operator fun Batch<IOType.D1>.plus(other: Float): Batch<IOType.D1> {
        val result = Backend.plus(x = value, y = other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("1982")
    inline operator fun Batch<IOType.D1>.plus(other: Batch<IOType.D0>): Batch<IOType.D1> {
        val result = Backend.plus(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 0,
        )
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("8102")
    inline operator fun Batch<IOType.D1>.plus(other: IOType.D1): Batch<IOType.D1> {
        val result = Backend.plus(x = value, xi = size, xj = step, y = other.value, axis = 1)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("16477")
    inline operator fun Batch<IOType.D1>.plus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.plus(x = value, y = other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("11295")
    inline fun Batch<IOType.D1>.plus(other: Batch<IOType.D2>, axis: Int): Batch<IOType.D2> {
        val result = Backend.plus(
            x = value,
            xi = size,
            xj = shape[0],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            axis1 = 0,
            axis2 = axis + 1,
        )
        return Batch<IOType.D2>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }

    @JvmName("15883")
    inline operator fun Batch<IOType.D3>.plus(other: Float): Batch<IOType.D3> {
        val result = Backend.plus(x = value, y = other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("4151")
    inline operator fun Batch<IOType.D3>.plus(other: Batch<IOType.D0>): Batch<IOType.D3> {
        val result = Backend.plus(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("22885")
    inline fun Batch<IOType.D3>.plus(other: IOType.D1, axis: Int): Batch<IOType.D3> {
        val result = Backend.plus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            axis = axis + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("22629")
    inline fun Batch<IOType.D3>.plus(other: Batch<IOType.D1>, axis: Int): Batch<IOType.D3> {
        val result = Backend.plus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            axis1 = 0,
            axis2 = axis + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("15793")
    inline operator fun Batch<IOType.D3>.plus(other: IOType.D2): Batch<IOType.D3> =
        plus(other = other, axis1 = 1, axis2 = 2)

    @JvmName("3628")
    inline fun Batch<IOType.D3>.plus(other: IOType.D2, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.plus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            yi = other.shape[0],
            yj = other.shape[1],
            axis1 = axis1 + 1,
            axis2 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("2979")
    inline operator fun Batch<IOType.D3>.plus(other: Batch<IOType.D2>) = plus(other, axis1 = 1, axis2 = 2)

    @JvmName("22904")
    inline fun Batch<IOType.D3>.plus(other: Batch<IOType.D2>, axis1: Int, axis2: Int): Batch<IOType.D3> {
        val result = Backend.plus(
            x = value,
            xi = size,
            xj = shape[0],
            xk = shape[1],
            xl = shape[2],
            y = other.value,
            yi = other.size,
            yj = other.shape[0],
            yk = other.shape[1],
            axis1 = 0,
            axis2 = axis1 + 1,
            axis3 = axis2 + 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("7174")
    inline operator fun Batch<IOType.D3>.plus(other: IOType.D3): Batch<IOType.D3> {
        val result = Backend.plus(
            x = value,
            xi = size,
            xj = step,
            y = other.value,
            axis = 1,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("26367")
    inline operator fun Batch<IOType.D3>.plus(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.plus(x = value, y = other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("28056")
    inline fun Batch<IOType.D0>.ln(e: Float = 1e-7f): Batch<IOType.D0> {
        val result = Backend.ln(x = value, e = e)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5420")
    inline fun Batch<IOType.D1>.ln(e: Float = 1e-7f): Batch<IOType.D1> {
        val result = Backend.ln(x = value, e = e)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("408")
    inline fun Batch<IOType.D2>.ln(e: Float = 1e-7f): Batch<IOType.D2> {
        val result = Backend.ln(x = value, e = e)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("21903")
    inline fun Batch<IOType.D3>.ln(e: Float = 1e-7f): Batch<IOType.D3> {
        val result = Backend.ln(x = value, e = e)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3486")
    inline fun Batch<IOType.D1>.softmax(): Batch<IOType.D1> = map { it.softmax() }

    @JvmName("27904")
    inline fun Batch<IOType.D2>.softmax(): Batch<IOType.D2> = map { it.softmax() }

    @JvmName("12046")
    inline fun Batch<IOType.D2>.softmax(axis: Int): Batch<IOType.D2> = map { it.softmax(axis = axis) }

    @JvmName("9747")
    inline fun Batch<IOType.D3>.softmax(): Batch<IOType.D3> = map { it.softmax() }

    @JvmName("21429")
    inline fun Batch<IOType.D3>.softmax(axis: Int): Batch<IOType.D3> = map { it.softmax(axis = axis) }

    @JvmName("25805")
    inline fun Batch<IOType.D1>.exp(): Batch<IOType.D1> {
        val result = Backend.exp(x = value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("29701")
    inline fun Batch<IOType.D2>.exp(): Batch<IOType.D2> {
        val result = Backend.exp(x = value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("10558")
    inline fun Batch<IOType.D3>.exp(): Batch<IOType.D3> {
        val result = Backend.exp(x = value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("866")
    inline fun Batch<IOType.D0>.pow(n: Int): Batch<IOType.D0> {
        val result = Backend.pow(x = value, n = n)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("8048")
    inline fun Batch<IOType.D1>.pow(n: Int): Batch<IOType.D1> {
        val result = Backend.pow(x = value, n = n)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("18780")
    inline fun Batch<IOType.D2>.pow(n: Int): Batch<IOType.D2> {
        val result = Backend.pow(x = value, n = n)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("15961")
    inline fun Batch<IOType.D3>.pow(n: Int): Batch<IOType.D3> {
        val result = Backend.pow(x = value, n = n)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3370")
    inline fun Batch<IOType.D0>.sqrt(e: Float = 1e-7f): Batch<IOType.D0> {
        val result = Backend.sqrt(x = value, e = e)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("2713")
    inline fun Batch<IOType.D1>.sqrt(e: Float = 1e-7f): Batch<IOType.D1> {
        val result = Backend.sqrt(x = value, e = e)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("19814")
    inline fun Batch<IOType.D2>.sqrt(e: Float = 1e-7f): Batch<IOType.D2> {
        val result = Backend.sqrt(x = value, e = e)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("22833")
    inline fun Batch<IOType.D3>.sqrt(e: Float = 1e-7f): Batch<IOType.D3> {
        val result = Backend.sqrt(x = value, e = e)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("12183")
    inline fun Batch<IOType.D1>.sigmoid(): Batch<IOType.D1> =
        Batch<IOType.D1>(size = size, shape = shape, value = Backend.sigmoid(value)).also { register(it.value) }

    @JvmName("18993")
    inline fun Batch<IOType.D2>.sigmoid(): Batch<IOType.D2> =
        Batch<IOType.D2>(size = size, shape = shape, value = Backend.sigmoid(value)).also { register(it.value) }

    @JvmName("29817")
    inline fun Batch<IOType.D3>.sigmoid(): Batch<IOType.D3> =
        Batch<IOType.D3>(size = size, shape = shape, value = Backend.sigmoid(value)).also { register(it.value) }

    @JvmName("19464")
    inline fun Batch<IOType.D1>.max(): Batch<IOType.D0> {
        val result = Backend.max(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }

    @JvmName("20142")
    inline fun Batch<IOType.D2>.max(): Batch<IOType.D0> {
        val result = Backend.max(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }

    @JvmName("28279")
    inline fun Batch<IOType.D3>.max(): Batch<IOType.D0> {
        val result = Backend.max(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }

    @JvmName("22055")
    inline fun Batch<IOType.D1>.min(): Batch<IOType.D0> {
        val result = Backend.min(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }

    @JvmName("4439")
    inline fun Batch<IOType.D2>.min(): Batch<IOType.D0> {
        val result = Backend.min(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }

    @JvmName("3816")
    inline fun Batch<IOType.D3>.min(): Batch<IOType.D0> {
        val result = Backend.min(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }

    @JvmName("6495")
    inline fun Batch<IOType.D2>.map(block: (IOType.D2) -> IOType.D2): Batch<IOType.D2> =
        Batch<IOType.D2>(size) { block(this[it]).also { register(it.value) } }

    @JvmName("11418")
    inline fun Batch<IOType.D0>.map(block: (IOType.D0) -> IOType.D0): Batch<IOType.D0> =
        Batch(size) { block(this[it]).also { register(it.value) } }

    @JvmName("23919")
    inline fun Batch<IOType.D1>.map(block: (IOType.D1) -> IOType.D1): Batch<IOType.D1> =
        Batch(size) { block(this[it]).also { register(it.value) } }

    @JvmName("5896")
    inline fun Batch<IOType.D3>.map(block: (IOType.D3) -> IOType.D3): Batch<IOType.D3> =
        Batch(size) { block(this[it]).also { register(it.value) } }

    @JvmName("27073")
    inline fun Batch<IOType.D2>.sum(): Batch<IOType.D0> {
        val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(size = size, shape = listOf(1), value = result).also { register(it.value) }
    }

    @JvmName("29640")
    inline fun Batch<IOType.D2>.sum(axis: Int): Batch<IOType.D1> {
        val result = Backend.sum(x = value, xi = size, xj = shape[0], xk = shape[1], axis = axis + 1)
        return Batch<IOType.D1>(
            size = size,
            shape = listOf(if (axis == 0) shape[1] else shape[0]),
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("3791")
    inline fun Batch<IOType.D1>.sum(): Batch<IOType.D0> {
        val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }

    @JvmName("24487")
    inline fun Batch<IOType.D3>.sum(): Batch<IOType.D0> {
        val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(size = size, shape = listOf(1), value = result).also { register(it.value) }
    }

    @JvmName("25580")
    inline fun Batch<IOType.D3>.sum(axis: Int): Batch<IOType.D2> = when (axis) {
        0 -> Batch(
            size = size,
            shape = listOf(shape[1], shape[2]),
            value = Backend.sum(x = value, xi = size, xj = shape[0], xk = shape[1] * shape[2], axis = 1),
        )

        1 -> Batch(
            size = size,
            shape = listOf(shape[0], shape[2]),
            value = Backend.sum(x = value, xi = size * shape[0], xj = shape[1], xk = shape[2], axis = 1),
        )

        2 -> Batch(
            size = size,
            shape = listOf(shape[0], shape[1]),
            value = Backend.sum(x = value, xi = size, xj = shape[0] * shape[1], xk = shape[2], axis = 2),
        )

        else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
    }

    @JvmName("16426")
    inline fun Batch<IOType.D4>.batchAverage(): IOType.D4 {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("18240")
    inline fun Batch<IOType.D2>.average(): Batch<IOType.D0> {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(size = size, shape = listOf(1), value = result).also { register(it.value) }
    }

    @JvmName("28907")
    inline fun Batch<IOType.D2>.average(axis: Int): Batch<IOType.D1> {
        val result = Backend.average(x = value, xi = size, xj = shape[0], xk = shape[1], axis = axis + 1)
        return Batch<IOType.D1>(
            size = size,
            shape = when (axis) {
                0 -> listOf(shape[1])
                else -> listOf(shape[0])
            },
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("21346")
    inline fun Batch<IOType.D2>.batchAverage(): IOType.D2 {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("26677")
    inline fun Batch<IOType.D0>.batchAverage(): IOType.D0 =
        IOType.D0(value = Backend.average(value)).also { register(it.value) }

    @JvmName("22015")
    inline fun Batch<IOType.D1>.average(): Batch<IOType.D0> {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(size = size, shape = listOf(1), value = result).also { register(it.value) }
    }

    @JvmName("16478")
    inline fun Batch<IOType.D1>.batchAverage(): IOType.D1 {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("30885")
    inline fun Batch<IOType.D3>.average(): Batch<IOType.D0> {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(size = size, shape = listOf(1), value = result).also { register(it.value) }
    }

    @JvmName("23485")
    inline fun Batch<IOType.D3>.average(axis: Int): Batch<IOType.D2> = when (axis) {
        0 -> Batch<IOType.D2>(
            size = size,
            shape = listOf(shape[1], shape[2]),
            value = Backend.average(x = value, xi = size, xj = shape[0], xk = shape[1] * shape[2], axis = 1),
        ).also { register(it.value) }

        1 -> Batch<IOType.D2>(
            size = size,
            shape = listOf(shape[0], shape[2]),
            value = Backend.average(x = value, xi = size * shape[0], xj = shape[1], xk = shape[2], axis = 1),
        ).also { register(it.value) }

        2 -> Batch<IOType.D2>(
            size = size,
            shape = listOf(shape[0], shape[1]),
            value = Backend.average(x = value, xi = size, xj = shape[0] * shape[1], xk = shape[2], axis = 2),
        ).also { register(it.value) }

        else -> throw IllegalArgumentException("axis is $axis, not 0, 1 or 2.")
    }

    @JvmName("22899")
    inline fun Batch<IOType.D3>.batchAverage(): IOType.D3 {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("30029")
    inline fun <T : IOType> List<T>.toBatch(): Batch<T> {
        val batchSize = size
        val shape = first().shape
        val step = shape.reduce { acc, i -> acc * i }
        val batchValue = DataBuffer.create(batchSize * step)
        forEachIndexed { index, item ->
            val start = index * step
            Backend.copyInto(item.value, batchValue, start until start + item.value.size)
        }
        register(batchValue)
        return Batch(
            value = batchValue,
            size = batchSize,
            shape = shape,
        )
    }

    @JvmName("28990")
    inline fun Batch<IOType.D1>.toList(): List<IOType.D1> = List(size) { get(it) }

    @JvmName("8017")
    inline fun Batch<IOType.D2>.toList(): List<IOType.D2> = List(size) { get(it) }

    @JvmName("9445")
    inline fun Batch<IOType.D3>.toList(): List<IOType.D3> = List(size) { get(it) }

    @JvmName("6794")
    inline infix fun Batch<IOType.D4>.lt(other: Float): Batch<IOType.D4> {
        val result = Backend.lessThan(value, other)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("25747")
    inline infix fun Batch<IOType.D4>.lt(other: Batch<IOType.D4>): Batch<IOType.D4> {
        val result = Backend.lessThan(value, other.value)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("9785")
    inline infix fun Batch<IOType.D2>.lt(other: Float): Batch<IOType.D2> {
        val result = Backend.lessThan(value, other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13870")
    inline infix fun Batch<IOType.D2>.lt(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.lessThan(value, other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("31929")
    inline infix fun Batch<IOType.D0>.lt(other: Float): Batch<IOType.D0> {
        val result = Backend.lessThan(value, other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("8543")
    inline infix fun Batch<IOType.D0>.lt(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.lessThan(value, other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3075")
    inline infix fun Batch<IOType.D1>.lt(other: Float): Batch<IOType.D1> {
        val result = Backend.lessThan(value, other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("14863")
    inline infix fun Batch<IOType.D1>.lt(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.lessThan(value, other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("12425")
    inline infix fun Batch<IOType.D3>.lt(other: Float): Batch<IOType.D3> {
        val result = Backend.lessThan(value, other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("18276")
    inline infix fun Batch<IOType.D3>.lt(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.lessThan(value, other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("14610")
    inline infix fun Batch<IOType.D4>.gt(other: Float): Batch<IOType.D4> {
        val result = Backend.greaterThan(value, other)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13022")
    inline infix fun Batch<IOType.D4>.gt(other: Batch<IOType.D4>): Batch<IOType.D4> {
        val result = Backend.greaterThan(value, other.value)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("32162")
    inline infix fun Batch<IOType.D2>.gt(other: Float): Batch<IOType.D2> {
        val result = Backend.greaterThan(value, other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13240")
    inline infix fun Batch<IOType.D2>.gt(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.greaterThan(value, other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("29908")
    inline infix fun Batch<IOType.D0>.gt(other: Float): Batch<IOType.D0> {
        val result = Backend.greaterThan(value, other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("1960")
    inline infix fun Batch<IOType.D0>.gt(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.greaterThan(value, other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("29918")
    inline infix fun Batch<IOType.D1>.gt(other: Float): Batch<IOType.D1> {
        val result = Backend.greaterThan(value, other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("4779")
    inline infix fun Batch<IOType.D1>.gt(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.greaterThan(value, other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("14609")
    inline infix fun Batch<IOType.D3>.gt(other: Float): Batch<IOType.D3> {
        val result = Backend.greaterThan(value, other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("6788")
    inline infix fun Batch<IOType.D3>.gt(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.greaterThan(value, other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("19086")
    inline infix fun Batch<IOType.D4>.eq(other: Float): Batch<IOType.D4> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("27693")
    inline fun Batch<IOType.D4>.eq(
        other: Float,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): Batch<IOType.D4> {
        val result = Backend.equals(
            x = value,
            y = other,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("9853")
    inline infix fun Batch<IOType.D4>.eq(other: Batch<IOType.D4>): Batch<IOType.D4> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("31374")
    inline fun Batch<IOType.D4>.eq(
        other: Batch<IOType.D4>,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): Batch<IOType.D4> {
        val result = Backend.equals(
            x = value,
            y = other.value,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("31650")
    inline infix fun Batch<IOType.D2>.eq(other: Float): Batch<IOType.D2> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("28077")
    inline fun Batch<IOType.D2>.eq(
        other: Float,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): Batch<IOType.D2> {
        val result = Backend.equals(
            x = value,
            y = other,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("31946")
    inline infix fun Batch<IOType.D2>.eq(other: Batch<IOType.D2>): Batch<IOType.D2> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("28556")
    inline fun Batch<IOType.D2>.eq(
        other: Batch<IOType.D2>,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): Batch<IOType.D2> {
        val result = Backend.equals(
            x = value,
            y = other.value,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("10721")
    inline infix fun Batch<IOType.D0>.eq(other: Float): Batch<IOType.D0> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("32459")
    inline fun Batch<IOType.D0>.eq(
        other: Float,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): Batch<IOType.D0> {
        val result = Backend.equals(
            x = value,
            y = other,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13933")
    inline infix fun Batch<IOType.D0>.eq(other: Batch<IOType.D0>): Batch<IOType.D0> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("9508")
    inline fun Batch<IOType.D0>.eq(
        other: Batch<IOType.D0>,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): Batch<IOType.D0> {
        val result = Backend.equals(
            x = value,
            y = other.value,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("24593")
    inline infix fun Batch<IOType.D1>.eq(other: Float): Batch<IOType.D1> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("7662")
    inline fun Batch<IOType.D1>.eq(
        other: Float,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): Batch<IOType.D1> {
        val result = Backend.equals(
            x = value,
            y = other,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("13579")
    inline infix fun Batch<IOType.D1>.eq(other: Batch<IOType.D1>): Batch<IOType.D1> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("27803")
    inline fun Batch<IOType.D1>.eq(
        other: Batch<IOType.D1>,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): Batch<IOType.D1> {
        val result = Backend.equals(
            x = value,
            y = other.value,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("19126")
    inline infix fun Batch<IOType.D3>.eq(other: Float): Batch<IOType.D3> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("25402")
    inline fun Batch<IOType.D3>.eq(
        other: Float,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): Batch<IOType.D3> {
        val result = Backend.equals(
            x = value,
            y = other,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("25609")
    inline infix fun Batch<IOType.D3>.eq(other: Batch<IOType.D3>): Batch<IOType.D3> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("6389")
    inline fun Batch<IOType.D3>.eq(
        other: Batch<IOType.D3>,
        absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
    ): Batch<IOType.D3> {
        val result = Backend.equals(
            x = value,
            y = other.value,
            absoluteTolerance = absoluteTolerance,
            relativeTolerance = relativeTolerance,
        )
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("2070")
    inline fun where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Float): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D4>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("28296")
    inline fun where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Batch<IOType.D4>): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D4>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("20412")
    inline fun where(condition: Batch<IOType.D4>, onTrue: Batch<IOType.D4>, onFalse: Float): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D4>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("2794")
    inline fun where(
        condition: Batch<IOType.D4>,
        onTrue: Batch<IOType.D4>,
        onFalse: Batch<IOType.D4>,
    ): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D4>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("30773")
    inline fun Batch<IOType.D4>.where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Float): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("23494")
    inline fun Batch<IOType.D4>.where(
        onTrue: Float,
        onFalse: Float,
        condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
    ): Batch<IOType.D4> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("6361")
    inline fun Batch<IOType.D4>.where(
        condition: Batch<IOType.D4>,
        onTrue: Float,
        onFalse: Batch<IOType.D4> = this,
    ): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("20260")
    inline fun Batch<IOType.D4>.where(
        onTrue: Float,
        onFalse: Batch<IOType.D4> = this,
        condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
    ): Batch<IOType.D4> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("2596")
    inline fun Batch<IOType.D4>.where(
        condition: Batch<IOType.D4>,
        onTrue: Batch<IOType.D4> = this,
        onFalse: Float,
    ): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("16327")
    inline fun Batch<IOType.D4>.where(
        onTrue: Batch<IOType.D4> = this,
        onFalse: Float,
        condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("16653")
    inline fun Batch<IOType.D4>.where(
        condition: Batch<IOType.D4>,
        onTrue: Batch<IOType.D4> = this,
        onFalse: Batch<IOType.D4> = this,
    ): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("18159")
    inline fun Batch<IOType.D4>.where(
        onTrue: Batch<IOType.D4> = this,
        onFalse: Batch<IOType.D4> = this,
        condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("26485")
    inline fun where(condition: Batch<IOType.D2>, onTrue: Float, onFalse: Float): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D2>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("3708")
    inline fun where(condition: Batch<IOType.D2>, onTrue: Float, onFalse: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D2>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("23830")
    inline fun where(condition: Batch<IOType.D2>, onTrue: Batch<IOType.D2>, onFalse: Float): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D2>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("4467")
    inline fun where(
        condition: Batch<IOType.D2>,
        onTrue: Batch<IOType.D2>,
        onFalse: Batch<IOType.D2>,
    ): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D2>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("15588")
    inline fun Batch<IOType.D2>.where(condition: Batch<IOType.D2>, onTrue: Float, onFalse: Float): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("25622")
    inline fun Batch<IOType.D2>.where(
        onTrue: Float,
        onFalse: Float,
        condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("28539")
    inline fun Batch<IOType.D2>.where(
        condition: Batch<IOType.D2>,
        onTrue: Float,
        onFalse: Batch<IOType.D2> = this,
    ): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("21799")
    inline fun Batch<IOType.D2>.where(
        onTrue: Float,
        onFalse: Batch<IOType.D2> = this,
        condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("18921")
    inline fun Batch<IOType.D2>.where(
        condition: Batch<IOType.D2>,
        onTrue: Batch<IOType.D2> = this,
        onFalse: Float,
    ): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("9271")
    inline fun Batch<IOType.D2>.where(
        onTrue: Batch<IOType.D2> = this,
        onFalse: Float,
        condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("31296")
    inline fun Batch<IOType.D2>.where(
        condition: Batch<IOType.D2>,
        onTrue: Batch<IOType.D2> = this,
        onFalse: Batch<IOType.D2> = this,
    ): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5766")
    inline fun Batch<IOType.D2>.where(
        onTrue: Batch<IOType.D2> = this,
        onFalse: Batch<IOType.D2> = this,
        condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("12753")
    inline fun where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Float): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D0>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("8432")
    inline fun where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D0>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("30577")
    inline fun where(condition: Batch<IOType.D0>, onTrue: Batch<IOType.D0>, onFalse: Float): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D0>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("4163")
    inline fun where(
        condition: Batch<IOType.D0>,
        onTrue: Batch<IOType.D0>,
        onFalse: Batch<IOType.D0>,
    ): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D0>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("7481")
    inline fun Batch<IOType.D0>.where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Float): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("5444")
    inline fun Batch<IOType.D0>.where(
        onTrue: Float,
        onFalse: Float,
        condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
    ): Batch<IOType.D0> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("28081")
    inline fun Batch<IOType.D0>.where(
        condition: Batch<IOType.D0>,
        onTrue: Float,
        onFalse: Batch<IOType.D0> = this,
    ): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("24339")
    inline fun Batch<IOType.D0>.where(
        onTrue: Float,
        onFalse: Batch<IOType.D0> = this,
        condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
    ): Batch<IOType.D0> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("20299")
    inline fun Batch<IOType.D0>.where(
        condition: Batch<IOType.D0>,
        onTrue: Batch<IOType.D0> = this,
        onFalse: Float,
    ): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("4425")
    inline fun Batch<IOType.D0>.where(
        onTrue: Batch<IOType.D0> = this,
        onFalse: Float,
        condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("13008")
    inline fun Batch<IOType.D0>.where(
        condition: Batch<IOType.D0>,
        onTrue: Batch<IOType.D0> = this,
        onFalse: Batch<IOType.D0> = this,
    ): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("16066")
    inline fun Batch<IOType.D0>.where(
        onTrue: Batch<IOType.D0> = this,
        onFalse: Batch<IOType.D0> = this,
        condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("4688")
    inline fun where(condition: Batch<IOType.D1>, onTrue: Float, onFalse: Float): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D1>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("12450")
    inline fun where(condition: Batch<IOType.D1>, onTrue: Float, onFalse: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D1>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("13016")
    inline fun where(condition: Batch<IOType.D1>, onTrue: Batch<IOType.D1>, onFalse: Float): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D1>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("6348")
    inline fun where(
        condition: Batch<IOType.D1>,
        onTrue: Batch<IOType.D1>,
        onFalse: Batch<IOType.D1>,
    ): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D1>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("20014")
    inline fun Batch<IOType.D1>.where(condition: Batch<IOType.D1>, onTrue: Float, onFalse: Float): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("22175")
    inline fun Batch<IOType.D1>.where(
        onTrue: Float,
        onFalse: Float,
        condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("17339")
    inline fun Batch<IOType.D1>.where(
        condition: Batch<IOType.D1>,
        onTrue: Float,
        onFalse: Batch<IOType.D1> = this,
    ): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("28958")
    inline fun Batch<IOType.D1>.where(
        onTrue: Float,
        onFalse: Batch<IOType.D1> = this,
        condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("3078")
    inline fun Batch<IOType.D1>.where(
        condition: Batch<IOType.D1>,
        onTrue: Batch<IOType.D1> = this,
        onFalse: Float,
    ): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("22326")
    inline fun Batch<IOType.D1>.where(
        onTrue: Batch<IOType.D1> = this,
        onFalse: Float,
        condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("22088")
    inline fun Batch<IOType.D1>.where(
        condition: Batch<IOType.D1>,
        onTrue: Batch<IOType.D1> = this,
        onFalse: Batch<IOType.D1> = this,
    ): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("23997")
    inline fun Batch<IOType.D1>.where(
        onTrue: Batch<IOType.D1> = this,
        onFalse: Batch<IOType.D1> = this,
        condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("2672")
    inline fun where(condition: Batch<IOType.D3>, onTrue: Float, onFalse: Float): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D3>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("15581")
    inline fun where(condition: Batch<IOType.D3>, onTrue: Float, onFalse: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D3>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("15825")
    inline fun where(condition: Batch<IOType.D3>, onTrue: Batch<IOType.D3>, onFalse: Float): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D3>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("4011")
    inline fun where(
        condition: Batch<IOType.D3>,
        onTrue: Batch<IOType.D3>,
        onFalse: Batch<IOType.D3>,
    ): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D3>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("2077")
    inline fun Batch<IOType.D3>.where(condition: Batch<IOType.D3>, onTrue: Float, onFalse: Float): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("29757")
    inline fun Batch<IOType.D3>.where(
        onTrue: Float,
        onFalse: Float,
        condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("21543")
    inline fun Batch<IOType.D3>.where(
        condition: Batch<IOType.D3>,
        onTrue: Float,
        onFalse: Batch<IOType.D3> = this,
    ): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("10937")
    inline fun Batch<IOType.D3>.where(
        onTrue: Float,
        onFalse: Batch<IOType.D3> = this,
        condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("14477")
    inline fun Batch<IOType.D3>.where(
        condition: Batch<IOType.D3>,
        onTrue: Batch<IOType.D3> = this,
        onFalse: Float,
    ): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("19103")
    inline fun Batch<IOType.D3>.where(
        onTrue: Batch<IOType.D3> = this,
        onFalse: Float,
        condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("30239")
    inline fun Batch<IOType.D3>.where(
        condition: Batch<IOType.D3>,
        onTrue: Batch<IOType.D3> = this,
        onFalse: Batch<IOType.D3> = this,
    ): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("20721")
    inline fun Batch<IOType.D3>.where(
        onTrue: Batch<IOType.D3> = this,
        onFalse: Batch<IOType.D3> = this,
        condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("31557")
    inline fun IOType.D0.gather(other: Batch<IOType.D2>, axis: Int = 1): Batch<IOType.D1> = when (axis) {
        0 -> {
            val result =
                Backend.gather(x = value, y = other.value, i = other.size, j = other.shape[0], k = other.shape[1])
            Batch<IOType.D1>(
                size = other.size,
                shape = listOf(other.shape[1]),
                value = result,
            ).also { register(it.value) }
        }

        else -> {
            val result = Backend.gather(
                x = value,
                y = other.value,
                i = other.size * other.shape[0],
                j = other.shape[1],
                k = 1,
            )
            Batch<IOType.D1>(
                size = other.size,
                shape = listOf(other.shape[0]),
                value = result,
            ).also { register(it.value) }
        }
    }

    @JvmName("24943")
    inline fun Batch<IOType.D1>.gather(other: IOType.D2): Batch<IOType.D2> {
        val result = Backend.gather(x = value, y = other.value, i = 1, j = other.i, k = other.j)
        return Batch<IOType.D2>(
            size = size,
            shape = listOf(shape[0], other.j),
            value = result,
        ).also { register(it.value) }
    }

    @JvmName("8427")
    inline fun Batch<IOType.D2>.scatterAdd(other: Batch<IOType.D1>, n: Int): IOType.D2 {
        val result = Backend.scatterAdd(x = value, y = other.value, i = 1, j = n, k = shape[1], b = size)
        return IOType.D2(shape = listOf(n, shape[1]), value = result).also { register(it.value) }
    }

    @JvmName("13440")
    inline fun <T : IOType> Batch(size: Int, init: (index: Int) -> T): Batch<T> {
        val first = init(0)
        val value = DataBuffer.create(size * first.value.size)
        Backend.copyInto(first.value, value, first.value.indices)
        for (i in 1 until size) {
            val src = init(i).value
            val start = i * first.value.size
            Backend.copyInto(src, value, start until start + src.size)
        }
        register(value)
        return Batch(
            value = value,
            size = size,
            shape = first.shape,
        )
    }

    @JvmName("15949")
    inline fun <T : IOType> batchOf(vararg elements: T): Batch<T> {
        val batchSize = elements.size
        val shape = elements.first().shape
        val step = shape.reduce { acc, i -> acc * i }
        val batchValue = DataBuffer.create(batchSize * step)
        elements.forEachIndexed { index, item ->
            val start = index * step
            Backend.copyInto(item.value, batchValue, start until start + item.value.size)
        }
        register(batchValue)
        return Batch(
            value = batchValue,
            size = batchSize,
            shape = shape,
        )
    }

    @JvmName("24722")
    inline operator fun Batch<IOType.D0>.get(i: Int): IOType.D0 {
        val index = i * step
        return IOType.d0(value[index]).also { register(it.value) }
    }

    @JvmName("11149")
    inline operator fun Batch<IOType.D1>.get(i: Int): IOType.D1 {
        val index = i * step
        val result = Backend.slice(x = value, indices = index until index + step)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("12747")
    inline operator fun Batch<IOType.D2>.get(i: Int): IOType.D2 {
        val index = i * step
        val result = Backend.slice(x = value, indices = index until index + step)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("3949")
    inline operator fun Batch<IOType.D3>.get(i: Int): IOType.D3 {
        val index = i * step
        val result = Backend.slice(x = value, indices = index until index + step)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("24305")
    inline operator fun Batch<IOType.D4>.get(i: Int): IOType.D4 {
        val index = i * step
        val result = Backend.slice(x = value, indices = index until index + step)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("14949")
    inline operator fun Batch<IOType.D0>.set(i: Int, element: IOType.D0) {
        value[i] = element.value[0]
    }

    @JvmName("22494")
    inline operator fun Batch<IOType.D1>.set(i: Int, element: IOType.D1) {
        val start = i * step
        Backend.copyInto(element.value, value, start until start + element.value.size)
    }

    @JvmName("19925")
    inline operator fun Batch<IOType.D2>.set(i: Int, element: IOType.D2) {
        val start = i * step
        Backend.copyInto(element.value, value, start until start + element.value.size)
    }

    @JvmName("30259")
    inline operator fun Batch<IOType.D3>.set(i: Int, element: IOType.D3) {
        val start = i * step
        Backend.copyInto(element.value, value, start until start + element.value.size)
    }

    @JvmName("27450")
    inline operator fun Batch<IOType.D4>.set(i: Int, element: IOType.D4) {
        val start = i * step
        Backend.copyInto(element.value, value, start until start + element.value.size)
    }

    companion object {
        inline fun launch(block: IOScope.() -> Unit) {
            IOScope().use { scope -> scope.block() }
        }

        inline fun <T : IOType> launch(block: IOScope.() -> T): T = IOScope()
            .use { scope ->
                scope.block().also { scope.remove(it.value) }
            }

        inline fun <T : Batch<*>> launch(block: IOScope.() -> T): T = IOScope()
            .use { scope ->
                scope.block().also { scope.remove(it.value) }
            }

        inline fun <T : IOType> IOScope.launch(block: IOScope.() -> T): T = IOScope()
            .use { scope ->
                scope.block()
                    .also { scope.remove(it.value) }
                    .also { this.register(it.value) }
            }
    }
}
