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

    @JvmName("TEMP")
inline fun IOType.D1.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))

    @JvmName("TEMP")
inline fun IOType.D1.reshapeToD2(shape: List<Int>) =
        IOType.D2(shape = shape, value = value).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.reshapeToD3(i: Int, j: Int, k: Int) = reshapeToD3(shape = listOf(i, j, k))

    @JvmName("TEMP")
inline fun IOType.D2.reshapeToD3(shape: List<Int>) =
        IOType.D3(shape = shape, value = value).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.reshapeToD4(i: Int, j: Int, k: Int, l: Int) =
        IOType.D4(shape = listOf(i, j, k, l), value = value).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D3.reshapeToD2(i: Int, j: Int) =
        IOType.D2(shape = listOf(i, j), value = value).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D4.reshapeToD2(i: Int, j: Int) =
        IOType.D2(shape = listOf(i, j), value = value).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D4.reshapeToD3(i: Int, j: Int, k: Int) =
        IOType.D3(shape = listOf(i, j, k), value = value).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D1.slice(indices: IntProgression): IOType.D1 {
        val result = Backend.slice(x = value, indices = indices)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.D2.transpose(): IOType.D2 {
        val result = Backend.transpose(x = value, xi = i, xj = j)
        return IOType.D2(shape = listOf(j, i), value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.transpose(axisI: Int, axisJ: Int, axisK: Int): IOType.D3 {
        val result = Backend.transpose(x = value, xi = i, xj = j, xk = k, axisI = axisI, axisJ = axisJ, axisK = axisK)
        return IOType.D3(shape = listOf(shape[axisI], shape[axisJ], shape[axisK]), value = result)
            .also { register(it.value) }
    }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.D1.broadcastToD2(axis: Int, size: Int) = when (axis) {
        0 -> IOType.d2(size, shape[0]) { x, y -> this[y] }.also { register(it.value) }
        1 -> IOType.d2(shape[0], size) { x, y -> this[x] }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D1.broadcastToD2 axis is $axis not 0 or 1.")
    }

    @JvmName("TEMP")
inline fun IOType.D2.broadcastToD3(axis: Int, size: Int) = when (axis) {
        0 -> IOType.d3(size, shape[0], shape[1]) { i, j, k -> this[j, k] }.also { register(it.value) }
        1 -> IOType.d3(shape[0], size, shape[1]) { i, j, k -> this[i, k] }.also { register(it.value) }
        2 -> IOType.d3(shape[0], shape[1], size) { i, j, k -> this[i, j] }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D2.broadcastToD3 axis is $axis not 0, 1 or 2.")
    }

    @JvmName("TEMP")
inline fun IOType.D1.interleave(other: IOType.D1): IOType.D1 {
        check(size == other.size)

        val result = DataBuffer.create(size + other.size)
        Backend.copyInto(x = value, y = result, indices = 0 until size * 2 step 2)
        Backend.copyInto(x = other.value, y = result, indices = 1 until size * 2 step 2)

        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    inline operator fun IOType.D4.div(other: Float): IOType.D4 {
        val result = Backend.div(x = value, y = other)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D4.div(other: IOType.D4): IOType.D4 {
        val result = Backend.div(x = value, y = other.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.div(other: Float): IOType.D2 {
        val result = Backend.div(x = value, y = other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.div(other: IOType.D0): IOType.D2 {
        val result = Backend.div(x = value, y = other.get())
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.div(other: IOType.D1, axis: Int): IOType.D2 {
        val result = Backend.div(x = value, xi = i, xj = j, y = other.value, axis = axis)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.div(other: IOType.D2): IOType.D2 {
        val result = Backend.div(x = value, y = other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    inline operator fun Float.div(other: IOType.D0): IOType.D0 =
        IOType.d0(this / other.get()).also { register(it.value) }

    inline operator fun Float.div(other: IOType.D1): IOType.D1 {
        val result = Backend.div(x = this, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun Float.div(other: IOType.D2): IOType.D2 {
        val result = Backend.div(x = this, y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun Float.div(other: IOType.D3): IOType.D3 {
        val result = Backend.div(x = this, y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun Float.div(other: IOType.D4): IOType.D4 {
        val result = Backend.div(x = this, y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.div(other: Float): IOType.D0 = IOType.d0(get() / other).also { register(it.value) }

    inline operator fun IOType.D0.div(other: IOType.D0): IOType.D0 =
        IOType.d0(get() / other.get()).also { register(it.value) }

    inline operator fun IOType.D0.div(other: IOType.D1): IOType.D1 {
        val result = Backend.div(x = get(), y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.div(other: IOType.D2): IOType.D2 {
        val result = Backend.div(x = get(), y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.div(other: IOType.D3): IOType.D3 {
        val result = Backend.div(x = get(), y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.div(other: IOType.D4): IOType.D4 {
        val result = Backend.div(x = get(), y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.div(other: Float): IOType.D1 {
        val result = Backend.div(x = value, y = other)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.div(other: IOType.D0): IOType.D1 {
        val result = Backend.div(x = value, y = other.get())
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.div(other: IOType.D1): IOType.D1 {
        val result = Backend.div(x = value, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.div(other: IOType.D2, axis: Int): IOType.D2 {
        val result = Backend.div(x = value, y = other.value, yi = other.i, yj = other.j, axis = axis)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D3.div(other: Float): IOType.D3 {
        val result = Backend.div(x = value, y = other)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D3.div(other: IOType.D0): IOType.D3 {
        val result = Backend.div(x = value, y = other.get())
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
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

    inline operator fun IOType.D3.div(other: IOType.D3): IOType.D3 {
        val result = Backend.div(x = value, y = other.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D4.minus(other: Float): IOType.D4 {
        val result = Backend.minus(x = value, y = other)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D4.minus(other: IOType.D4): IOType.D4 {
        val result = Backend.minus(x = value, y = other.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.minus(other: Float): IOType.D2 {
        val result = Backend.minus(x = value, y = other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.minus(other: IOType.D0): IOType.D2 {
        val result = Backend.minus(x = value, y = other.get())
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.minus(other: IOType.D1, axis: Int): IOType.D2 {
        val result = Backend.minus(x = value, xi = i, xj = j, y = other.value, axis = axis)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.minus(other: IOType.D2): IOType.D2 {
        val result = Backend.minus(x = value, y = other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    inline operator fun Float.minus(other: IOType.D0): IOType.D0 =
        IOType.d0(value = this - other.get()).also { register(it.value) }

    inline operator fun Float.minus(other: IOType.D1): IOType.D1 {
        val result = Backend.minus(x = this, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun Float.minus(other: IOType.D2): IOType.D2 {
        val result = Backend.minus(x = this, y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun Float.minus(other: IOType.D3): IOType.D3 {
        val result = Backend.minus(x = this, y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun Float.minus(other: IOType.D4): IOType.D4 {
        val result = Backend.minus(x = this, y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.minus(other: Float): IOType.D0 =
        IOType.d0(value = get() - other).also { register(it.value) }

    inline operator fun IOType.D0.minus(other: IOType.D0): IOType.D0 =
        IOType.d0(get() - other.get()).also { register(it.value) }

    inline operator fun IOType.D0.minus(other: IOType.D1): IOType.D1 {
        val result = Backend.minus(x = get(), y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.minus(other: IOType.D2): IOType.D2 {
        val result = Backend.minus(x = get(), y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.minus(other: IOType.D3): IOType.D3 {
        val result = Backend.minus(x = get(), y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.minus(other: IOType.D4): IOType.D4 {
        val result = Backend.minus(x = get(), y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.minus(other: Float): IOType.D1 {
        val result = Backend.minus(x = value, y = other)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.minus(other: IOType.D0): IOType.D1 {
        val result = Backend.minus(x = value, y = other.get())
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.minus(other: IOType.D1): IOType.D1 {
        val result = Backend.minus(x = value, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.minus(other: IOType.D2, axis: Int): IOType.D2 {
        val result = Backend.minus(x = value, y = other.value, yi = other.i, yj = other.j, axis = axis)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D3.minus(other: Float): IOType.D3 {
        val result = Backend.minus(x = value, y = other)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D3.minus(other: IOType.D0): IOType.D3 {
        val result = Backend.minus(x = value, y = other.get())
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
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

    inline operator fun IOType.D3.minus(other: IOType.D3): IOType.D3 {
        val result = Backend.minus(x = value, y = other.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D1.inner(other: IOType.D1): Float = Backend.inner(x = value, y = other.value, b = 1)[0]

    inline operator fun IOType.D4.times(other: Float): IOType.D4 {
        val result = Backend.times(x = value, y = other)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D4.times(other: IOType.D4): IOType.D4 {
        val result = Backend.times(x = value, y = other.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.times(other: Float): IOType.D2 {
        val result = Backend.times(x = value, y = other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.times(other: IOType.D0): IOType.D2 {
        val result = Backend.times(x = value, y = other.get())
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.times(other: IOType.D1, axis: Int): IOType.D2 {
        val result = Backend.times(x = value, xi = i, xj = j, y = other.value, axis = axis)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.times(other: IOType.D2): IOType.D2 {
        val result = Backend.times(x = value, y = other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    inline operator fun Float.times(other: IOType.D0): IOType.D0 =
        IOType.d0(this * other.get()).also { register(it.value) }

    inline operator fun Float.times(other: IOType.D1): IOType.D1 {
        val result = Backend.times(x = this, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun Float.times(other: IOType.D2): IOType.D2 {
        val result = Backend.times(x = this, y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun Float.times(other: IOType.D3): IOType.D3 {
        val result = Backend.times(x = this, y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun Float.times(other: IOType.D4): IOType.D4 {
        val result = Backend.times(x = this, y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.times(other: Float): IOType.D0 = IOType.d0(get() * other).also { register(it.value) }

    inline operator fun IOType.D0.times(other: IOType.D0): IOType.D0 =
        IOType.d0(get() * other.get()).also { register(it.value) }

    inline operator fun IOType.D0.times(other: IOType.D1): IOType.D1 {
        val result = Backend.times(x = get(), y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.times(other: IOType.D2): IOType.D2 {
        val result = Backend.times(x = get(), y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.times(other: IOType.D3): IOType.D3 {
        val result = Backend.times(x = get(), y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.times(other: IOType.D4): IOType.D4 {
        val result = Backend.times(x = get(), y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.times(other: Float): IOType.D1 {
        val result = Backend.times(x = value, y = other)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.times(other: IOType.D0): IOType.D1 {
        val result = Backend.times(x = value, y = other.get())
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.times(other: IOType.D1): IOType.D1 {
        val result = Backend.times(x = value, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.times(other: IOType.D2, axis: Int): IOType.D2 {
        val result = Backend.times(x = value, y = other.value, yi = other.i, yj = other.j, axis = axis)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D3.times(other: Float): IOType.D3 {
        val result = Backend.times(x = value, y = other)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D3.times(other: IOType.D0): IOType.D3 {
        val result = Backend.times(x = value, y = other.get())
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
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

    inline operator fun IOType.D3.times(other: IOType.D3): IOType.D3 {
        val result = Backend.times(x = value, y = other.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.zipWith(other: IOType.D1, axis: Int, block: (Float, Float) -> Float): IOType.D2 = when (axis) {
        0 -> IOType.d2(shape) { i, j -> block(this[i, j], other[i]) }.also { register(it.value) }
        1 -> IOType.d2(shape) { i, j -> block(this[i, j], other[j]) }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D2.zipWith axis is $axis not 0 or 1.")
    }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.D1.zipWith(other: IOType.D2, axis: Int, block: (Float, Float) -> Float): IOType.D2 = when (axis) {
        0 -> IOType.d2(other.shape) { i, j -> block(this[i], other[i, j]) }.also { register(it.value) }
        1 -> IOType.d2(other.shape) { i, j -> block(this[j], other[i, j]) }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D1.zipWith axis is $axis not 0 or 1.")
    }

    @JvmName("TEMP")
inline fun IOType.D1.zipWith(other: IOType.D3, axis: Int, block: (Float, Float) -> Float): IOType.D3 = when (axis) {
        0 -> IOType.d3(shape) { i, j, k -> block(this[i], other[i, j, k]) }.also { register(it.value) }
        1 -> IOType.d3(shape) { i, j, k -> block(this[j], other[i, j, k]) }.also { register(it.value) }
        2 -> IOType.d3(shape) { i, j, k -> block(this[k], other[i, j, k]) }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D3.zipWith axis is $axis not 0, 1 or 2.")
    }

    @JvmName("TEMP")
inline fun IOType.D3.zipWith(other: IOType.D1, axis: Int, block: (Float, Float) -> Float): IOType.D3 = when (axis) {
        0 -> IOType.d3(shape) { i, j, k -> block(this[i, j, k], other[i]) }.also { register(it.value) }
        1 -> IOType.d3(shape) { i, j, k -> block(this[i, j, k], other[j]) }.also { register(it.value) }
        2 -> IOType.d3(shape) { i, j, k -> block(this[i, j, k], other[k]) }.also { register(it.value) }
        else -> throw IllegalArgumentException("IOType.D3.zipWith axis is $axis not 0, 1 or 2.")
    }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
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

    @JvmName("TEMP")
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

    @JvmName("TEMP")
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

    inline operator fun IOType.D4.plus(other: Float): IOType.D4 {
        val result = Backend.plus(x = value, y = other)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D4.plus(other: IOType.D4): IOType.D4 {
        val result = Backend.plus(x = value, y = other.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.plus(other: Float): IOType.D2 {
        val result = Backend.plus(x = value, y = other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.plus(other: IOType.D0): IOType.D2 {
        val result = Backend.plus(x = value, y = other.get())
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.plus(other: IOType.D1, axis: Int): IOType.D2 {
        val result = Backend.plus(x = value, xi = i, xj = j, y = other.value, axis = axis)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D2.plus(other: IOType.D2): IOType.D2 {
        val result = Backend.plus(x = value, y = other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    inline operator fun Float.plus(other: IOType.D0): IOType.D0 =
        IOType.d0(value = this + other.get()).also { register(it.value) }

    inline operator fun Float.plus(other: IOType.D1): IOType.D1 {
        val result = Backend.plus(x = this, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun Float.plus(other: IOType.D2): IOType.D2 {
        val result = Backend.plus(x = this, y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun Float.plus(other: IOType.D3): IOType.D3 {
        val result = Backend.plus(x = this, y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun Float.plus(other: IOType.D4): IOType.D4 {
        val result = Backend.plus(x = this, y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.plus(other: Float): IOType.D0 =
        IOType.d0(value = get() + other).also { register(it.value) }

    inline operator fun IOType.D0.plus(other: IOType.D0): IOType.D0 =
        IOType.d0(get() + other.get()).also { register(it.value) }

    inline operator fun IOType.D0.plus(other: IOType.D1): IOType.D1 {
        val result = Backend.plus(x = get(), y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.plus(other: IOType.D2): IOType.D2 {
        val result = Backend.plus(x = get(), y = other.value)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.plus(other: IOType.D3): IOType.D3 {
        val result = Backend.plus(x = get(), y = other.value)
        return IOType.D3(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D0.plus(other: IOType.D4): IOType.D4 {
        val result = Backend.plus(x = get(), y = other.value)
        return IOType.D4(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.plus(other: Float): IOType.D1 {
        val result = Backend.plus(x = value, y = other)
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.plus(other: IOType.D0): IOType.D1 {
        val result = Backend.plus(x = value, y = other.get())
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D1.plus(other: IOType.D1): IOType.D1 {
        val result = Backend.plus(x = value, y = other.value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.plus(other: IOType.D2, axis: Int): IOType.D2 {
        val result = Backend.plus(x = value, y = other.value, yi = other.i, yj = other.j, axis = axis)
        return IOType.D2(shape = other.shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D3.plus(other: Float): IOType.D3 {
        val result = Backend.plus(x = value, y = other)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun IOType.D3.plus(other: IOType.D0): IOType.D3 {
        val result = Backend.plus(x = value, y = other.get())
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
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

    inline operator fun IOType.D3.plus(other: IOType.D3): IOType.D3 {
        val result = Backend.plus(x = value, y = other.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.max(): IOType.D0 = IOType.D0(Backend.max(x = value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.max(): IOType.D0 = IOType.D0(Backend.max(x = value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.max(axis: Int): IOType.D1 {
        val result = Backend.max(x = value, xi = i, xj = j, axis = axis)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.max(): IOType.D0 = IOType.D0(Backend.max(x = value)).also { register(it.value) }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.D1.sum(): IOType.D0 = IOType.D0(Backend.sum(x = value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.sum(): IOType.D0 = IOType.D0(Backend.sum(x = value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.sum(axis: Int): IOType.D1 {
        val result = Backend.sum(x = value, xi = i, xj = j, axis = axis)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.sum(): IOType.D0 = IOType.D0(Backend.sum(value)).also { register(it.value) }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.D4.sum(): IOType.D0 = IOType.D0(Backend.sum(value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D1.topK(k: Int, random: Random = Random): Int = Backend.topK(x = value, k = k, random = random)

    @JvmName("TEMP")
inline fun IOType.D1.topP(p: Float, random: Random = Random): Int = Backend.topP(x = value, p = p, random = random)

    @JvmName("TEMP")
inline fun IOType.D1.maxIndex(): Int = Backend.maxIndex(x = value)

    @JvmName("TEMP")
inline fun IOType.D1.min() = IOType.D0(Backend.min(x = value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.min() = IOType.D0(Backend.min(x = value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.min(axis: Int): IOType.D1 {
        val result = Backend.min(x = value, xi = i, xj = j, axis = axis)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.min() = IOType.D0(Backend.min(x = value)).also { register(it.value) }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.D1.average(): IOType.D0 = IOType.D0(Backend.average(value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.average(): IOType.D0 = IOType.D0(Backend.average(value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.average(axis: Int): IOType.D1 {
        val result = Backend.average(x = value, xi = i, xj = j, axis = axis)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.average(): IOType.D0 = IOType.D0(Backend.average(value)).also { register(it.value) }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.D1.ln(e: Float): IOType.D1 {
        val result = Backend.ln(x = value, e = e)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.ln(e: Float): IOType.D2 {
        val result = Backend.ln(x = value, e = e)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.ln(e: Float): IOType.D3 {
        val result = Backend.ln(x = value, e = e)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.softmax(): IOType.D1 {
        val max = max()
        val exp = (this - max).exp()
        val sum = exp.sum()
        return (exp / sum).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.softmax(): IOType.D2 {
        val max = max()
        val exp = (this - max).exp()
        val sum = exp.sum()
        return (exp / sum).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.softmax(axis: Int): IOType.D2 {
        val max = max(axis = axis)
        val exp = this.minus(other = max, axis = if (axis == 0) 1 else 0).exp()
        val sum = exp.sum(axis = axis)
        return exp.div(other = sum, axis = if (axis == 0) 1 else 0).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.softmax(): IOType.D3 {
        val max = max()
        val exp = (this - max).exp()
        val sum = exp.sum()
        return (exp / sum).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.D1.exp(): IOType.D1 {
        val result = Backend.exp(x = value)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.exp(): IOType.D2 {
        val result = Backend.exp(x = value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.exp(): IOType.D3 {
        val result = Backend.exp(x = value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D0.pow(n: Int): IOType.D0 {
        val result = Backend.pow(x = value, n = n)
        return IOType.D0(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.pow(n: Int): IOType.D1 {
        val result = Backend.pow(x = value, n = n)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.pow(n: Int): IOType.D2 {
        val result = Backend.pow(x = value, n = n)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.pow(n: Int): IOType.D3 {
        val result = Backend.pow(x = value, n = n)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D4.pow(n: Int): IOType.D4 {
        val result = Backend.pow(x = value, n = n)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D0.sqrt(e: Float = 1e-7f): IOType.D0 {
        val result = Backend.sqrt(x = value, e = e)
        return IOType.D0(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.sqrt(e: Float = 1e-7f): IOType.D1 {
        val result = Backend.sqrt(x = value, e = e)
        return IOType.D1(value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.sqrt(e: Float = 1e-7f): IOType.D2 {
        val result = Backend.sqrt(x = value, e = e)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.sqrt(e: Float = 1e-7f): IOType.D3 {
        val result = Backend.sqrt(x = value, e = e)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D4.sqrt(e: Float = 1e-7f): IOType.D4 {
        val result = Backend.sqrt(x = value, e = e)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D0.gt(other: Float): IOType.D0 {
        val result = Backend.greaterThan(value, other)
        return IOType.D0(result).also { register(it.value) }
    }

    inline infix fun IOType.D0.gt(other: IOType.D0): IOType.D0 {
        val result = Backend.greaterThan(value, other.value)
        return IOType.D0(result).also { register(it.value) }
    }

    inline infix fun IOType.D1.gt(other: Float): IOType.D1 {
        val result = Backend.greaterThan(value, other)
        return IOType.D1(result).also { register(it.value) }
    }

    inline infix fun IOType.D1.gt(other: IOType.D1): IOType.D1 {
        val result = Backend.greaterThan(value, other.value)
        return IOType.D1(result).also { register(it.value) }
    }

    inline infix fun IOType.D2.gt(other: Float): IOType.D2 {
        val result = Backend.greaterThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D2.gt(other: IOType.D2): IOType.D2 {
        val result = Backend.greaterThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D3.gt(other: Float): IOType.D2 {
        val result = Backend.greaterThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D3.gt(other: IOType.D2): IOType.D2 {
        val result = Backend.greaterThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D4.gt(other: Float): IOType.D2 {
        val result = Backend.greaterThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D4.gt(other: IOType.D2): IOType.D2 {
        val result = Backend.greaterThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D0.lt(other: Float): IOType.D0 {
        val result = Backend.lessThan(value, other)
        return IOType.D0(result).also { register(it.value) }
    }

    inline infix fun IOType.D0.lt(other: IOType.D0): IOType.D0 {
        val result = Backend.lessThan(value, other.value)
        return IOType.D0(result).also { register(it.value) }
    }

    inline infix fun IOType.D1.lt(other: Float): IOType.D1 {
        val result = Backend.lessThan(value, other)
        return IOType.D1(result).also { register(it.value) }
    }

    inline infix fun IOType.D1.lt(other: IOType.D1): IOType.D1 {
        val result = Backend.lessThan(value, other.value)
        return IOType.D1(result).also { register(it.value) }
    }

    inline infix fun IOType.D2.lt(other: Float): IOType.D2 {
        val result = Backend.lessThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D2.lt(other: IOType.D2): IOType.D2 {
        val result = Backend.lessThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D3.lt(other: Float): IOType.D2 {
        val result = Backend.lessThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D3.lt(other: IOType.D2): IOType.D2 {
        val result = Backend.lessThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D4.lt(other: Float): IOType.D2 {
        val result = Backend.lessThan(value, other)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D4.lt(other: IOType.D2): IOType.D2 {
        val result = Backend.lessThan(value, other.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    inline infix fun IOType.D0.eq(other: Float) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("TEMP")
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

    inline infix fun IOType.D0.eq(other: IOType.D0) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("TEMP")
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

    inline infix fun IOType.D1.eq(other: Float) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("TEMP")
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

    inline infix fun IOType.D1.eq(other: IOType.D1) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("TEMP")
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

    inline infix fun IOType.D2.eq(other: Float) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("TEMP")
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

    inline infix fun IOType.D2.eq(other: IOType.D2) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("TEMP")
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

    inline infix fun IOType.D3.eq(other: Float) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("TEMP")
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

    inline infix fun IOType.D3.eq(other: IOType.D3) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("TEMP")
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

    inline infix fun IOType.D4.eq(other: Float) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("TEMP")
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

    inline infix fun IOType.D4.eq(other: IOType.D4) = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun where(condition: IOType.D4, onTrue: Float, onFalse: Float): IOType.D4 {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return IOType.D4(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D4, onTrue: Float, onFalse: IOType.D4): IOType.D4 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D4(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D4, onTrue: IOType.D4, onFalse: Float): IOType.D4 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D4(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D4, onTrue: IOType.D4, onFalse: IOType.D4): IOType.D4 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D4(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D4.where(onTrue: Float, onFalse: Float, condition: (IOType.D4) -> IOType.D4) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D4.where(condition: IOType.D4, onTrue: Float, onFalse: IOType.D4 = this): IOType.D4 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D4.where(onTrue: Float, onFalse: IOType.D4 = this, condition: (IOType.D4) -> IOType.D4) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D4.where(condition: IOType.D4, onTrue: IOType.D4 = this, onFalse: Float): IOType.D4 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D4.where(onTrue: IOType.D4 = this, onFalse: Float, condition: (IOType.D4) -> IOType.D4) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D4.where(condition: IOType.D4, onTrue: IOType.D4 = this, onFalse: IOType.D4 = this): IOType.D4 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D4.where(
        onTrue: IOType.D4 = this,
        onFalse: IOType.D4 = this,
        condition: (IOType.D4) -> IOType.D4,
    ) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun where(condition: IOType.D2, onTrue: Float, onFalse: Float): IOType.D2 {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return IOType.D2(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D2, onTrue: Float, onFalse: IOType.D2): IOType.D2 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D2(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D2, onTrue: IOType.D2, onFalse: Float): IOType.D2 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D2(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D2, onTrue: IOType.D2, onFalse: IOType.D2): IOType.D2 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D2(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.where(onTrue: Float, onFalse: Float, condition: (IOType.D2) -> IOType.D2) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D2.where(condition: IOType.D2, onTrue: Float, onFalse: IOType.D2 = this): IOType.D2 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.where(onTrue: Float, onFalse: IOType.D2 = this, condition: (IOType.D2) -> IOType.D2) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D2.where(condition: IOType.D2, onTrue: IOType.D2 = this, onFalse: Float): IOType.D2 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.where(onTrue: IOType.D2 = this, onFalse: Float, condition: (IOType.D2) -> IOType.D2) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D2.where(condition: IOType.D2, onTrue: IOType.D2 = this, onFalse: IOType.D2 = this): IOType.D2 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.where(
        onTrue: IOType.D2 = this,
        onFalse: IOType.D2 = this,
        condition: (IOType.D2) -> IOType.D2,
    ) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun where(condition: IOType.D0, onTrue: Float, onFalse: Float): IOType.D0 {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D0, onTrue: Float, onFalse: IOType.D0): IOType.D0 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D0, onTrue: IOType.D0, onFalse: Float): IOType.D0 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D0, onTrue: IOType.D0, onFalse: IOType.D0): IOType.D0 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D0.where(onTrue: Float, onFalse: Float, condition: (IOType.D0) -> IOType.D0) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D0.where(condition: IOType.D0, onTrue: Float, onFalse: IOType.D0 = this): IOType.D0 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D0.where(onTrue: Float, onFalse: IOType.D0, condition: (IOType.D0) -> IOType.D0) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D0.where(condition: IOType.D0, onTrue: IOType.D0 = this, onFalse: Float): IOType.D0 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D0.where(onTrue: IOType.D0 = this, onFalse: Float, condition: (IOType.D0) -> IOType.D0) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D0.where(condition: IOType.D0, onTrue: IOType.D0 = this, onFalse: IOType.D0 = this): IOType.D0 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D0(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D0.where(onTrue: IOType.D0 = this, onFalse: IOType.D0, condition: (IOType.D0) -> IOType.D0) =
        where(
            condition = condition(this),
            onTrue = onTrue,
            onFalse = onFalse,
        )

    @JvmName("TEMP")
inline fun where(condition: IOType.D1, onTrue: Float, onFalse: Float): IOType.D1 {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D1, onTrue: Float, onFalse: IOType.D1): IOType.D1 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D1, onTrue: IOType.D1, onFalse: Float): IOType.D1 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D1, onTrue: IOType.D1, onFalse: IOType.D1): IOType.D1 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.where(onTrue: Float, onFalse: Float, condition: (IOType.D1) -> IOType.D1) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D1.where(condition: IOType.D1, onTrue: Float, onFalse: IOType.D1 = this): IOType.D1 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.where(onTrue: Float, onFalse: IOType.D1, condition: (IOType.D1) -> IOType.D1) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D1.where(condition: IOType.D1, onTrue: IOType.D1 = this, onFalse: Float): IOType.D1 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.where(onTrue: IOType.D1 = this, onFalse: Float, condition: (IOType.D1) -> IOType.D1) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D1.where(condition: IOType.D1, onTrue: IOType.D1 = this, onFalse: IOType.D1 = this): IOType.D1 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D1(result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D1.where(onTrue: IOType.D1 = this, onFalse: IOType.D1, condition: (IOType.D1) -> IOType.D1) =
        where(
            condition = condition(this),
            onTrue = onTrue,
            onFalse = onFalse,
        )

    @JvmName("TEMP")
inline fun where(condition: IOType.D3, onTrue: Float, onFalse: Float): IOType.D3 {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return IOType.D3(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D3, onTrue: Float, onFalse: IOType.D3): IOType.D3 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D3(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D3, onTrue: IOType.D3, onFalse: Float): IOType.D3 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D3(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun where(condition: IOType.D3, onTrue: IOType.D3, onFalse: IOType.D3): IOType.D3 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D3(shape = condition.shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.where(onTrue: Float, onFalse: Float, condition: (IOType.D3) -> IOType.D3) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D3.where(condition: IOType.D3, onTrue: Float, onFalse: IOType.D3 = this): IOType.D3 {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.where(onTrue: Float, onFalse: IOType.D3 = this, condition: (IOType.D3) -> IOType.D3) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D3.where(condition: IOType.D3, onTrue: IOType.D3 = this, onFalse: Float): IOType.D3 {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.where(onTrue: IOType.D3 = this, onFalse: Float, condition: (IOType.D3) -> IOType.D3) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D3.where(condition: IOType.D3, onTrue: IOType.D3 = this, onFalse: IOType.D3 = this): IOType.D3 {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D3.where(
        onTrue: IOType.D3 = this,
        onFalse: IOType.D3 = this,
        condition: (IOType.D3) -> IOType.D3,
    ) = where(
        condition = condition(this),
        onTrue = onTrue,
        onFalse = onFalse,
    )

    @JvmName("TEMP")
inline fun IOType.D1.gather(other: IOType.D2): IOType.D2 {
        val result = Backend.gather(x = value, y = other.value, i = 1, j = other.i, k = other.j)
        return IOType.D2(shape = listOf(size, other.j), value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.D2.scatterAdd(other: IOType.D1, n: Int): IOType.D2 {
        val result = Backend.scatterAdd(x = value, y = other.value, i = 1, j = n, k = j, b = 1)
        return IOType.D2(shape = listOf(n, j), value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.Companion.d0(value: Float) = IOType.d0(floatArrayOf(value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.Companion.d0(value: FloatArray) = IOType.D0(DataBuffer.create(value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.Companion.d1(size: Int, init: (Int) -> Float = { 0f }): IOType.D1 {
        val value = FloatArray(size)
        for (i in 0 until size) value[i] = init(i)
        return IOType.d1(value = value).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.Companion.d1(shape: List<Int>, init: (Int) -> Float = { 0f }) =
        d1(shape[0], init).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.Companion.d1(value: List<Float>) =
        IOType.d1(value = value.toFloatArray()).also { register(it.value) }


    @JvmName("TEMP")
inline fun IOType.Companion.d1(vararg elements: Float) = IOType.d1(value = elements).also { register(it.value) }


    @JvmName("TEMP")
inline fun IOType.Companion.d1(value: FloatArray) =
        IOType.D1(value = DataBuffer.create(value)).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.Companion.d2(i: Int, j: Int, init: (Int, Int) -> Float = { _, _ -> 0f }): IOType.D2 {
        val value = FloatArray(i * j)
        for (_i in 0 until i) {
            for (_j in 0 until j) {
                value[_i * j + _j] = init(_i, _j)
            }
        }
        return IOType.d2(shape = listOf(i, j), value = value).also { register(it.value) }
    }

    @JvmName("TEMP")
inline fun IOType.Companion.d2(shape: List<Int>, init: (Int, Int) -> Float = { _, _ -> 0f }) = d2(
        i = shape[0],
        j = shape[1],
        init = init,
    ).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.Companion.d2(shape: List<Int>, value: List<Float>) = IOType.d2(
        value = value.toFloatArray(),
        shape = shape,
    ).also { register(it.value) }


    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.Companion.d2(shape: List<Int>, value: FloatArray) =
        IOType.D2(shape = shape, value = DataBuffer.create(value)).also { register(it.value) }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.Companion.d3(shape: List<Int>, init: (Int, Int, Int) -> Float = { _, _, _ -> 0f }) = d3(
        i = shape[0],
        j = shape[1],
        k = shape[2],
        init = init,
    ).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.Companion.d3(shape: List<Int>, value: List<Float>) = IOType.d3(
        value = value.toFloatArray(),
        shape = shape,
    ).also { register(it.value) }


    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.Companion.d3(shape: List<Int>, value: FloatArray) =
        IOType.D3(shape = shape, value = DataBuffer.create(value)).also { register(it.value) }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.Companion.d4(shape: List<Int>, init: (Int, Int, Int, Int) -> Float = { _, _, _, _ -> 0f }) = d4(
        i = shape[0],
        j = shape[1],
        k = shape[2],
        l = shape[3],
        init = init,
    ).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.Companion.d4(shape: List<Int>, value: List<Float>) = IOType.d4(
        value = value.toFloatArray(),
        shape = shape,
    ).also { register(it.value) }


    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun IOType.Companion.d4(shape: List<Int>, value: FloatArray) =
        IOType.D4(shape = shape, value = DataBuffer.create(value)).also { register(it.value) }

    /**
     * get
     */
    @JvmName("TEMP")
inline fun IOType.D0.get() = value[0]

    inline operator fun IOType.D1.get(index: Int) = value[index]

    inline operator fun IOType.D2.get(i: Int, j: Int) = value[i * shape[1] + j]

    inline operator fun IOType.D2.get(i: Int): IOType.D1 {
        val offset = i * shape[1]
        val result = Backend.slice(x = value, indices = offset until offset + shape[1])
        return IOType.D1(result).also { register(it.value) }
    }

    inline operator fun IOType.D3.get(i: Int, j: Int, k: Int) = value[(i * shape[1] + j) * shape[2] + k]

    inline operator fun IOType.D3.get(i: Int, j: Int): IOType.D1 {
        val offset = (i * shape[1] + j) * shape[2]
        val result = Backend.slice(x = value, indices = offset until offset + shape[2])
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D3.get(i: Int): IOType.D2 {
        val offset = i * shape[1] * shape[2]
        val result = Backend.slice(x = value, indices = offset until offset + shape[1] * shape[2])
        return IOType.D2(
            shape = listOf(shape[1], shape[2]),
            value = result,
        ).also { register(it.value) }
    }

    inline operator fun IOType.D4.get(i: Int, j: Int, k: Int, l: Int) =
        value[((i * shape[1] + j) * shape[2] + k) * shape[3] + l]

    inline operator fun IOType.D4.get(i: Int, j: Int, k: Int): IOType.D1 {
        val offset = ((i * shape[1] + j) * shape[2] + k) * shape[3]
        val result = Backend.slice(x = value, indices = offset until offset + shape[3])
        return IOType.D1(value = result).also { register(it.value) }
    }

    inline operator fun IOType.D4.get(i: Int, j: Int): IOType.D2 {
        val offset = (i * shape[1] + j) * shape[2] * shape[3]
        val result = Backend.slice(x = value, indices = offset until offset + shape[2] * shape[3])
        return IOType.D2(
            shape = listOf(shape[2], shape[3]),
            value = result,
        ).also { register(it.value) }
    }

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
    @JvmName("TEMP")
inline fun IOType.D0.set(element: Float) {
        value[0] = element
    }

    inline operator fun IOType.D1.set(index: Int, element: Float) {
        value[index] = element
    }

    inline operator fun IOType.D2.set(i: Int, j: Int, element: Float) {
        value[i * shape[1] + j] = element
    }

    inline operator fun IOType.D2.set(i: Int, element: IOType.D1) {
        val start = i * shape[1]
        Backend.copyInto(
            x = element.value,
            y = value,
            indices = start until start + element.value.size,
        )
    }

    inline operator fun IOType.D3.set(i: Int, j: Int, z: Int, element: Float) {
        value[(i * shape[1] + j) * shape[2] + z] = element
    }

    inline operator fun IOType.D3.set(i: Int, j: Int, element: IOType.D1) {
        val start = (i * shape[1] + j) * shape[2]
        Backend.copyInto(
            x = element.value,
            y = value,
            indices = start until start + element.value.size,
        )
    }

    inline operator fun IOType.D3.set(i: Int, element: IOType.D2) {
        val start = i * shape[1] * shape[2]
        Backend.copyInto(
            x = element.value,
            y = value,
            indices = start until start + element.value.size,
        )
    }

    inline operator fun IOType.D4.set(i: Int, j: Int, k: Int, l: Int, element: Float) {
        value[((i * shape[1] + j) * shape[2] + k) * shape[3] + l] = element
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.reshapeToD2(shape: List<Int>) =
        Batch<IOType.D2>(size = size, shape = shape, value = value)


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.reshapeToD3(i: Int, j: Int, k: Int) = reshapeToD3(listOf(i, j, k))


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.reshapeToD3(shape: List<Int>) =
        Batch<IOType.D3>(size = size, shape = shape, value = value)


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.reshapeToD3(i: Int, j: Int, k: Int) = reshapeToD3(listOf(i, j, k))


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.reshapeToD3(shape: List<Int>) =
        Batch<IOType.D3>(size = size, shape = shape, value = value)


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.reshapeToD2(i: Int, j: Int) = reshapeToD2(listOf(i, j))


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.reshapeToD2(shape: List<Int>) =
        Batch<IOType.D2>(size = size, shape = shape, value = value)


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.slice(indices: IntProgression): Batch<IOType.D1> {
        val result = Backend.slice(x = value, xi = size, xj = shape[0], axis = 1, indices = indices)
        return Batch<IOType.D1>(size = size, shape = listOf(indices.size), value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.transpose(): Batch<IOType.D2> {
        val result =
            Backend.transpose(x = value, xi = size, xj = shape[0], xk = shape[1], axisI = 0, axisJ = 2, axisK = 1)
        return Batch<IOType.D2>(size = size, shape = shape.reversed(), value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.flatten() = Batch<IOType.D1>(
        shape = listOf(step),
        size = size,
        value = value,
    )


    @JvmName("TEMP")
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


    @JvmName("TEMP")
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
    @JvmName("TEMP")
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


    @JvmName("TEMP")
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
    @JvmName("TEMP")
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

    @JvmName("TEMP")
inline fun Batch<IOType.D1>.broadcastToD2(axis: Int, size: Int) =
        Batch(this.size) { this[it].broadcastToD2(axis, size).also { register(it.value) } }

    @JvmName("TEMP")
inline fun Batch<IOType.D2>.broadcastToD3(axis: Int, size: Int) =
        Batch(this.size) { this[it].broadcastToD3(axis, size).also { register(it.value) } }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.toD2(): IOType.D2 =
        IOType.D2(shape = listOf(size, shape[0]), value = value).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D2.toD1(): Batch<IOType.D1> =
        Batch<IOType.D1>(value = value, size = shape[0], shape = listOf(shape[1])).also { register(it.value) }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.toD3(): IOType.D3 =
        IOType.D3(shape = listOf(size, shape[0], shape[1]), value = value).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D3.toBatch(): Batch<IOType.D2> =
        Batch<IOType.D2>(value = value, size = shape[0], shape = listOf(shape[1], shape[2])).also { register(it.value) }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.toD4(): IOType.D4 =
        IOType.D4(shape = listOf(size, shape[0], shape[1], shape[2]), value = value).also { register(it.value) }

    @JvmName("TEMP")
inline fun IOType.D4.toBatch(): Batch<IOType.D3> =
        Batch<IOType.D3>(value = value, size = shape[0], shape = listOf(shape[1], shape[2], shape[3])).also {
            register(
                it.value,
            )
        }


    inline operator fun Batch<IOType.D2>.div(other: Float): Batch<IOType.D2> {
        val result = Backend.div(x = value, y = other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D2>.div(other: Batch<IOType.D0>): Batch<IOType.D2> {
        val result = Backend.div(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.div(other: IOType.D1, axis: Int): Batch<IOType.D2> {
        val result = Backend.div(x = value, xi = size, xj = shape[0], xk = shape[1], y = other.value, axis = axis + 1)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D2>.div(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.div(x = value, y = other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Float.div(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.div(x = this, y = other.value)
        return Batch<IOType.D0>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.div(x = this, y = other.value)
        return Batch<IOType.D1>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.div(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.div(x = this, y = other.value)
        return Batch<IOType.D2>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.div(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.div(x = this, y = other.value)
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.div(other: Float): Batch<IOType.D0> {
        val result = Backend.div(x = value, y = other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.div(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.div(x = value, y = other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.div(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
        return Batch<IOType.D1>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }


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


    inline operator fun Batch<IOType.D1>.div(other: Float): Batch<IOType.D1> {
        val result = Backend.div(x = value, y = other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


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


    inline operator fun Batch<IOType.D1>.div(other: IOType.D1): Batch<IOType.D1> {
        val result = Backend.div(x = value, xi = size, xj = step, y = other.value, axis = 1)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D1>.div(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.div(x = value, y = other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.div(other: Float): Batch<IOType.D3> {
        val result = Backend.div(x = value, y = other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D3>.div(other: Batch<IOType.D0>): Batch<IOType.D3> {
        val result = Backend.div(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.div(other: IOType.D2): Batch<IOType.D3> =
        div(other = other, axis1 = 1, axis2 = 2)


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.div(other: Batch<IOType.D2>) = div(other, axis1 = 1, axis2 = 2)


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.div(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.div(x = value, y = other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D2>.minus(other: Float): Batch<IOType.D2> {
        val result = Backend.minus(x = value, y = other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D2>.minus(other: Batch<IOType.D0>): Batch<IOType.D2> {
        val result = Backend.minus(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.minus(other: IOType.D1, axis: Int): Batch<IOType.D2> {
        val result = Backend.minus(x = value, xi = size, xj = shape[0], xk = shape[1], y = other.value, axis = axis + 1)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D2>.minus(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.minus(x = value, y = other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Float.minus(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.minus(x = this, y = other.value)
        return Batch<IOType.D0>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.minus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.minus(x = this, y = other.value)
        return Batch<IOType.D1>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.minus(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.minus(x = this, y = other.value)
        return Batch<IOType.D2>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.minus(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.minus(x = this, y = other.value)
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.minus(other: Float): Batch<IOType.D0> {
        val result = Backend.minus(x = value, y = other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.minus(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.minus(x = value, y = other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.minus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.minus(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
        return Batch<IOType.D1>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }


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


    inline operator fun Batch<IOType.D1>.minus(other: Float): Batch<IOType.D1> {
        val result = Backend.minus(x = value, y = other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


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


    inline operator fun Batch<IOType.D1>.minus(other: IOType.D1): Batch<IOType.D1> {
        val result = Backend.minus(x = value, xi = size, xj = step, y = other.value, axis = 1)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D1>.minus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.minus(x = value, y = other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.minus(other: Float): Batch<IOType.D3> {
        val result = Backend.minus(x = value, y = other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D3>.minus(other: Batch<IOType.D0>): Batch<IOType.D3> {
        val result = Backend.minus(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.minus(other: IOType.D2): Batch<IOType.D3> =
        minus(other = other, axis1 = 1, axis2 = 2)


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.minus(other: Batch<IOType.D2>) = minus(other, axis1 = 1, axis2 = 2)


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.minus(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.minus(x = value, y = other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D1>.inner(other: Batch<IOType.D1>): Batch<IOType.D0> {
        val result = Backend.inner(x = value, y = other.value, b = size)
        return Batch<IOType.D0>(value = result, size = size, shape = listOf(1)).also { register(it.value) }
    }


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


    inline operator fun Batch<IOType.D2>.times(other: Float): Batch<IOType.D2> {
        val result = Backend.times(x = value, y = other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D2>.times(other: Batch<IOType.D0>): Batch<IOType.D2> {
        val result = Backend.times(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.times(other: IOType.D1, axis: Int): Batch<IOType.D2> {
        val result = Backend.times(x = value, xi = size, xj = shape[0], xk = shape[1], y = other.value, axis = axis + 1)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D2>.times(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.times(x = value, y = other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Float.times(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.times(x = this, y = other.value)
        return Batch<IOType.D0>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.times(x = this, y = other.value)
        return Batch<IOType.D1>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.times(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.times(x = this, y = other.value)
        return Batch<IOType.D2>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.times(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.times(x = this, y = other.value)
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.times(other: Float): Batch<IOType.D0> {
        val result = Backend.times(x = value, y = other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.times(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.times(x = value, y = other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.times(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
        return Batch<IOType.D1>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }


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


    inline operator fun Batch<IOType.D1>.times(other: Float): Batch<IOType.D1> {
        val result = Backend.times(x = value, y = other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


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


    inline operator fun Batch<IOType.D1>.times(other: IOType.D1): Batch<IOType.D1> {
        val result = Backend.times(x = value, xi = size, xj = step, y = other.value, axis = 1)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D1>.times(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.times(x = value, y = other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.times(other: Float): Batch<IOType.D3> {
        val result = Backend.times(x = value, y = other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D3>.times(other: Batch<IOType.D0>): Batch<IOType.D3> {
        val result = Backend.times(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.times(other: IOType.D2): Batch<IOType.D3> =
        times(other = other, axis1 = 1, axis2 = 2)


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.times(other: Batch<IOType.D2>) = times(other, axis1 = 1, axis2 = 2)


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.times(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.times(x = value, y = other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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


    @JvmName("TEMP")
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


    @JvmName("TEMP")
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

    @JvmName("TEMP")
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


    @JvmName("TEMP")
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


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D2>.plus(other: Float): Batch<IOType.D2> {
        val result = Backend.plus(x = value, y = other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D2>.plus(other: Batch<IOType.D0>): Batch<IOType.D2> {
        val result = Backend.plus(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.plus(other: IOType.D1, axis: Int): Batch<IOType.D2> {
        val result = Backend.plus(x = value, xi = size, xj = shape[0], xk = shape[1], y = other.value, axis = axis + 1)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D2>.plus(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.plus(x = value, y = other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Float.plus(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.plus(x = this, y = other.value)
        return Batch<IOType.D0>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.plus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.plus(x = this, y = other.value)
        return Batch<IOType.D1>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.plus(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.plus(x = this, y = other.value)
        return Batch<IOType.D2>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Float.plus(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.plus(x = this, y = other.value)
        return Batch<IOType.D3>(size = other.size, shape = other.shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.plus(other: Float): Batch<IOType.D0> {
        val result = Backend.plus(x = value, y = other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.plus(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.plus(x = value, y = other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D0>.plus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.plus(x = value, y = other.value, yi = other.size, yj = other.step, axis = 0)
        return Batch<IOType.D1>(size = size, shape = other.shape, value = result).also { register(it.value) }
    }


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


    inline operator fun Batch<IOType.D1>.plus(other: Float): Batch<IOType.D1> {
        val result = Backend.plus(x = value, y = other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


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


    inline operator fun Batch<IOType.D1>.plus(other: IOType.D1): Batch<IOType.D1> {
        val result = Backend.plus(x = value, xi = size, xj = step, y = other.value, axis = 1)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D1>.plus(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.plus(x = value, y = other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.plus(other: Float): Batch<IOType.D3> {
        val result = Backend.plus(x = value, y = other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D3>.plus(other: Batch<IOType.D0>): Batch<IOType.D3> {
        val result = Backend.plus(x = value, xi = size, xj = step, y = other.value, axis = 0)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.plus(other: IOType.D2): Batch<IOType.D3> =
        plus(other = other, axis1 = 1, axis2 = 2)


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.plus(other: Batch<IOType.D2>) = plus(other, axis1 = 1, axis2 = 2)


    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D3>.plus(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.plus(x = value, y = other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.ln(e: Float = 1e-7f): Batch<IOType.D0> {
        val result = Backend.ln(x = value, e = e)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.ln(e: Float = 1e-7f): Batch<IOType.D1> {
        val result = Backend.ln(x = value, e = e)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.ln(e: Float = 1e-7f): Batch<IOType.D2> {
        val result = Backend.ln(x = value, e = e)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.ln(e: Float = 1e-7f): Batch<IOType.D3> {
        val result = Backend.ln(x = value, e = e)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.softmax(): Batch<IOType.D1> = map { it.softmax() }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.softmax(): Batch<IOType.D2> = map { it.softmax() }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.softmax(axis: Int): Batch<IOType.D2> = map { it.softmax(axis = axis) }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.softmax(): Batch<IOType.D3> = map { it.softmax() }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.softmax(axis: Int): Batch<IOType.D3> = map { it.softmax(axis = axis) }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.exp(): Batch<IOType.D1> {
        val result = Backend.exp(x = value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.exp(): Batch<IOType.D2> {
        val result = Backend.exp(x = value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.exp(): Batch<IOType.D3> {
        val result = Backend.exp(x = value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.pow(n: Int): Batch<IOType.D0> {
        val result = Backend.pow(x = value, n = n)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.pow(n: Int): Batch<IOType.D1> {
        val result = Backend.pow(x = value, n = n)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.pow(n: Int): Batch<IOType.D2> {
        val result = Backend.pow(x = value, n = n)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.pow(n: Int): Batch<IOType.D3> {
        val result = Backend.pow(x = value, n = n)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.sqrt(e: Float = 1e-7f): Batch<IOType.D0> {
        val result = Backend.sqrt(x = value, e = e)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.sqrt(e: Float = 1e-7f): Batch<IOType.D1> {
        val result = Backend.sqrt(x = value, e = e)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.sqrt(e: Float = 1e-7f): Batch<IOType.D2> {
        val result = Backend.sqrt(x = value, e = e)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.sqrt(e: Float = 1e-7f): Batch<IOType.D3> {
        val result = Backend.sqrt(x = value, e = e)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.sigmoid(): Batch<IOType.D1> =
        Batch<IOType.D1>(size = size, shape = shape, value = Backend.sigmoid(value)).also { register(it.value) }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.sigmoid(): Batch<IOType.D2> =
        Batch<IOType.D2>(size = size, shape = shape, value = Backend.sigmoid(value)).also { register(it.value) }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.sigmoid(): Batch<IOType.D3> =
        Batch<IOType.D3>(size = size, shape = shape, value = Backend.sigmoid(value)).also { register(it.value) }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.max(): Batch<IOType.D0> {
        val result = Backend.max(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.max(): Batch<IOType.D0> {
        val result = Backend.max(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.max(): Batch<IOType.D0> {
        val result = Backend.max(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.min(): Batch<IOType.D0> {
        val result = Backend.min(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.min(): Batch<IOType.D0> {
        val result = Backend.min(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.min(): Batch<IOType.D0> {
        val result = Backend.min(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.map(block: (IOType.D2) -> IOType.D2): Batch<IOType.D2> =
        Batch<IOType.D2>(size) { block(this[it]).also { register(it.value) } }


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.map(block: (IOType.D0) -> IOType.D0): Batch<IOType.D0> =
        Batch(size) { block(this[it]).also { register(it.value) } }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.map(block: (IOType.D1) -> IOType.D1): Batch<IOType.D1> =
        Batch(size) { block(this[it]).also { register(it.value) } }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.map(block: (IOType.D3) -> IOType.D3): Batch<IOType.D3> =
        Batch(size) { block(this[it]).also { register(it.value) } }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.sum(): Batch<IOType.D0> {
        val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(size = size, shape = listOf(1), value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.sum(axis: Int): Batch<IOType.D1> {
        val result = Backend.sum(x = value, xi = size, xj = shape[0], xk = shape[1], axis = axis + 1)
        return Batch<IOType.D1>(
            size = size,
            shape = listOf(if (axis == 0) shape[1] else shape[0]),
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.sum(): Batch<IOType.D0> {
        val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(shape = listOf(1), size = size, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.sum(): Batch<IOType.D0> {
        val result = Backend.sum(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(size = size, shape = listOf(1), value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D4>.batchAverage(): IOType.D4 {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.average(): Batch<IOType.D0> {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(size = size, shape = listOf(1), value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.batchAverage(): IOType.D2 {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.batchAverage(): IOType.D0 =
        IOType.D0(value = Backend.average(value)).also { register(it.value) }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.average(): Batch<IOType.D0> {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(size = size, shape = listOf(1), value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.batchAverage(): IOType.D1 {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
        return IOType.D1(value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.average(): Batch<IOType.D0> {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 1)
        return Batch<IOType.D0>(size = size, shape = listOf(1), value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.batchAverage(): IOType.D3 {
        val result = Backend.average(x = value, xi = size, xj = step, axis = 0)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.toList(): List<IOType.D1> = List(size) { get(it) }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.toList(): List<IOType.D2> = List(size) { get(it) }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.toList(): List<IOType.D3> = List(size) { get(it) }


    inline infix fun Batch<IOType.D4>.lt(other: Float): Batch<IOType.D4> {
        val result = Backend.lessThan(value, other)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D4>.lt(other: Batch<IOType.D4>): Batch<IOType.D4> {
        val result = Backend.lessThan(value, other.value)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D2>.lt(other: Float): Batch<IOType.D2> {
        val result = Backend.lessThan(value, other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D2>.lt(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.lessThan(value, other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D0>.lt(other: Float): Batch<IOType.D0> {
        val result = Backend.lessThan(value, other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D0>.lt(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.lessThan(value, other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D1>.lt(other: Float): Batch<IOType.D1> {
        val result = Backend.lessThan(value, other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D1>.lt(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.lessThan(value, other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D3>.lt(other: Float): Batch<IOType.D3> {
        val result = Backend.lessThan(value, other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D3>.lt(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.lessThan(value, other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D4>.gt(other: Float): Batch<IOType.D4> {
        val result = Backend.greaterThan(value, other)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D4>.gt(other: Batch<IOType.D4>): Batch<IOType.D4> {
        val result = Backend.greaterThan(value, other.value)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D2>.gt(other: Float): Batch<IOType.D2> {
        val result = Backend.greaterThan(value, other)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D2>.gt(other: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.greaterThan(value, other.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D0>.gt(other: Float): Batch<IOType.D0> {
        val result = Backend.greaterThan(value, other)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D0>.gt(other: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.greaterThan(value, other.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D1>.gt(other: Float): Batch<IOType.D1> {
        val result = Backend.greaterThan(value, other)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D1>.gt(other: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.greaterThan(value, other.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D3>.gt(other: Float): Batch<IOType.D3> {
        val result = Backend.greaterThan(value, other)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D3>.gt(other: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.greaterThan(value, other.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    inline infix fun Batch<IOType.D4>.eq(other: Float): Batch<IOType.D4> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )


    @JvmName("TEMP")
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


    inline infix fun Batch<IOType.D4>.eq(other: Batch<IOType.D4>): Batch<IOType.D4> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )


    @JvmName("TEMP")
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


    inline infix fun Batch<IOType.D2>.eq(other: Float): Batch<IOType.D2> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )


    @JvmName("TEMP")
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


    inline infix fun Batch<IOType.D2>.eq(other: Batch<IOType.D2>): Batch<IOType.D2> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )


    @JvmName("TEMP")
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


    inline infix fun Batch<IOType.D0>.eq(other: Float): Batch<IOType.D0> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )


    @JvmName("TEMP")
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


    inline infix fun Batch<IOType.D0>.eq(other: Batch<IOType.D0>): Batch<IOType.D0> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )


    @JvmName("TEMP")
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


    inline infix fun Batch<IOType.D1>.eq(other: Float): Batch<IOType.D1> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )


    @JvmName("TEMP")
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


    inline infix fun Batch<IOType.D1>.eq(other: Batch<IOType.D1>): Batch<IOType.D1> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )


    @JvmName("TEMP")
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


    inline infix fun Batch<IOType.D3>.eq(other: Float): Batch<IOType.D3> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )


    @JvmName("TEMP")
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


    inline infix fun Batch<IOType.D3>.eq(other: Batch<IOType.D3>): Batch<IOType.D3> = eq(
        other = other,
        absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
        relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
    )


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Float): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D4>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Batch<IOType.D4>): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D4>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D4>, onTrue: Batch<IOType.D4>, onFalse: Float): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D4>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D4>.where(condition: Batch<IOType.D4>, onTrue: Float, onFalse: Float): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D4>.where(
        onTrue: Float,
        onFalse: Float,
        condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
    ): Batch<IOType.D4> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D4>.where(
        condition: Batch<IOType.D4>,
        onTrue: Float,
        onFalse: Batch<IOType.D4> = this,
    ): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D4>.where(
        onTrue: Float,
        onFalse: Batch<IOType.D4> = this,
        condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
    ): Batch<IOType.D4> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D4>.where(
        condition: Batch<IOType.D4>,
        onTrue: Batch<IOType.D4> = this,
        onFalse: Float,
    ): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D4>.where(
        onTrue: Batch<IOType.D4> = this,
        onFalse: Float,
        condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D4>.where(
        condition: Batch<IOType.D4>,
        onTrue: Batch<IOType.D4> = this,
        onFalse: Batch<IOType.D4> = this,
    ): Batch<IOType.D4> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D4>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D4>.where(
        onTrue: Batch<IOType.D4> = this,
        onFalse: Batch<IOType.D4> = this,
        condition: (Batch<IOType.D4>) -> Batch<IOType.D4>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D2>, onTrue: Float, onFalse: Float): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D2>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D2>, onTrue: Float, onFalse: Batch<IOType.D2>): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D2>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D2>, onTrue: Batch<IOType.D2>, onFalse: Float): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D2>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.where(condition: Batch<IOType.D2>, onTrue: Float, onFalse: Float): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.where(
        onTrue: Float,
        onFalse: Float,
        condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.where(
        condition: Batch<IOType.D2>,
        onTrue: Float,
        onFalse: Batch<IOType.D2> = this,
    ): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.where(
        onTrue: Float,
        onFalse: Batch<IOType.D2> = this,
        condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D2> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.where(
        condition: Batch<IOType.D2>,
        onTrue: Batch<IOType.D2> = this,
        onFalse: Float,
    ): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.where(
        onTrue: Batch<IOType.D2> = this,
        onFalse: Float,
        condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.where(
        condition: Batch<IOType.D2>,
        onTrue: Batch<IOType.D2> = this,
        onFalse: Batch<IOType.D2> = this,
    ): Batch<IOType.D2> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D2>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.where(
        onTrue: Batch<IOType.D2> = this,
        onFalse: Batch<IOType.D2> = this,
        condition: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Float): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D0>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Batch<IOType.D0>): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D0>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D0>, onTrue: Batch<IOType.D0>, onFalse: Float): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D0>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.where(condition: Batch<IOType.D0>, onTrue: Float, onFalse: Float): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.where(
        onTrue: Float,
        onFalse: Float,
        condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
    ): Batch<IOType.D0> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.where(
        condition: Batch<IOType.D0>,
        onTrue: Float,
        onFalse: Batch<IOType.D0> = this,
    ): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.where(
        onTrue: Float,
        onFalse: Batch<IOType.D0> = this,
        condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
    ): Batch<IOType.D0> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.where(
        condition: Batch<IOType.D0>,
        onTrue: Batch<IOType.D0> = this,
        onFalse: Float,
    ): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.where(
        onTrue: Batch<IOType.D0> = this,
        onFalse: Float,
        condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.where(
        condition: Batch<IOType.D0>,
        onTrue: Batch<IOType.D0> = this,
        onFalse: Batch<IOType.D0> = this,
    ): Batch<IOType.D0> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D0>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D0>.where(
        onTrue: Batch<IOType.D0> = this,
        onFalse: Batch<IOType.D0> = this,
        condition: (Batch<IOType.D0>) -> Batch<IOType.D0>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D1>, onTrue: Float, onFalse: Float): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D1>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D1>, onTrue: Float, onFalse: Batch<IOType.D1>): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D1>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D1>, onTrue: Batch<IOType.D1>, onFalse: Float): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D1>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.where(condition: Batch<IOType.D1>, onTrue: Float, onFalse: Float): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.where(
        onTrue: Float,
        onFalse: Float,
        condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.where(
        condition: Batch<IOType.D1>,
        onTrue: Float,
        onFalse: Batch<IOType.D1> = this,
    ): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.where(
        onTrue: Float,
        onFalse: Batch<IOType.D1> = this,
        condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ): Batch<IOType.D1> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.where(
        condition: Batch<IOType.D1>,
        onTrue: Batch<IOType.D1> = this,
        onFalse: Float,
    ): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.where(
        onTrue: Batch<IOType.D1> = this,
        onFalse: Float,
        condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.where(
        condition: Batch<IOType.D1>,
        onTrue: Batch<IOType.D1> = this,
        onFalse: Batch<IOType.D1> = this,
    ): Batch<IOType.D1> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D1>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.where(
        onTrue: Batch<IOType.D1> = this,
        onFalse: Batch<IOType.D1> = this,
        condition: (Batch<IOType.D1>) -> Batch<IOType.D1>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D3>, onTrue: Float, onFalse: Float): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D3>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D3>, onTrue: Float, onFalse: Batch<IOType.D3>): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D3>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun where(condition: Batch<IOType.D3>, onTrue: Batch<IOType.D3>, onFalse: Float): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D3>(
            size = condition.size,
            shape = condition.shape,
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.where(condition: Batch<IOType.D3>, onTrue: Float, onFalse: Float): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue, onFalse)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.where(
        onTrue: Float,
        onFalse: Float,
        condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.where(
        condition: Batch<IOType.D3>,
        onTrue: Float,
        onFalse: Batch<IOType.D3> = this,
    ): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue, onFalse.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.where(
        onTrue: Float,
        onFalse: Batch<IOType.D3> = this,
        condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ): Batch<IOType.D3> = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.where(
        condition: Batch<IOType.D3>,
        onTrue: Batch<IOType.D3> = this,
        onFalse: Float,
    ): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue.value, onFalse)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.where(
        onTrue: Batch<IOType.D3> = this,
        onFalse: Float,
        condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.where(
        condition: Batch<IOType.D3>,
        onTrue: Batch<IOType.D3> = this,
        onFalse: Batch<IOType.D3> = this,
    ): Batch<IOType.D3> {
        val result = Backend.where(condition.value, onTrue.value, onFalse.value)
        return Batch<IOType.D3>(size = size, shape = shape, value = result).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D3>.where(
        onTrue: Batch<IOType.D3> = this,
        onFalse: Batch<IOType.D3> = this,
        condition: (Batch<IOType.D3>) -> Batch<IOType.D3>,
    ) = where(onTrue = onTrue, onFalse = onFalse, condition = condition(this))

    @JvmName("TEMP")
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


    @JvmName("TEMP")
inline fun Batch<IOType.D1>.gather(other: IOType.D2): Batch<IOType.D2> {
        val result = Backend.gather(x = value, y = other.value, i = 1, j = other.i, k = other.j)
        return Batch<IOType.D2>(
            size = size,
            shape = listOf(shape[0], other.j),
            value = result,
        ).also { register(it.value) }
    }


    @JvmName("TEMP")
inline fun Batch<IOType.D2>.scatterAdd(other: Batch<IOType.D1>, n: Int): IOType.D2 {
        val result = Backend.scatterAdd(x = value, y = other.value, i = 1, j = n, k = shape[1], b = size)
        return IOType.D2(shape = listOf(n, shape[1]), value = result).also { register(it.value) }
    }

    @JvmName("TEMP")
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

    @JvmName("TEMP")
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


    inline operator fun Batch<IOType.D0>.get(i: Int): IOType.D0 {
        val index = i * step
        return IOType.d0(value[index]).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D1>.get(i: Int): IOType.D1 {
        val index = i * step
        val result = Backend.slice(x = value, indices = index until index + step)
        return IOType.D1(result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D2>.get(i: Int): IOType.D2 {
        val index = i * step
        val result = Backend.slice(x = value, indices = index until index + step)
        return IOType.D2(shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D3>.get(i: Int): IOType.D3 {
        val index = i * step
        val result = Backend.slice(x = value, indices = index until index + step)
        return IOType.D3(shape = shape, value = result).also { register(it.value) }
    }


    inline operator fun Batch<IOType.D4>.get(i: Int): IOType.D4 {
        val index = i * step
        val result = Backend.slice(x = value, indices = index until index + step)
        return IOType.D4(shape = shape, value = result).also { register(it.value) }
    }

    inline operator fun Batch<IOType.D0>.set(i: Int, element: IOType.D0) {
        value[i] = element.value[0]
    }

    inline operator fun Batch<IOType.D1>.set(i: Int, element: IOType.D1) {
        val start = i * step
        Backend.copyInto(element.value, value, start until start + element.value.size)
    }

    inline operator fun Batch<IOType.D2>.set(i: Int, element: IOType.D2) {
        val start = i * step
        Backend.copyInto(element.value, value, start until start + element.value.size)
    }

    inline operator fun Batch<IOType.D3>.set(i: Int, element: IOType.D3) {
        val start = i * step
        Backend.copyInto(element.value, value, start until start + element.value.size)
    }

    inline operator fun Batch<IOType.D4>.set(i: Int, element: IOType.D4) {
        val start = i * step
        Backend.copyInto(element.value, value, start until start + element.value.size)
    }

    companion object {
        @JvmName("TEMP")
inline fun launch(block: IOScope.() -> Unit) {
            IOScope().use { scope -> scope.block() }
        }

        @JvmName("TEMP")
inline fun <T : IOType> launch(block: IOScope.() -> T): T = IOScope()
            .use { scope ->
                scope.block().also { scope.remove(it.value) }
            }

        @JvmName("TEMP")
inline fun <T : Batch<*>> launch(block: IOScope.() -> T): T = IOScope()
            .use { scope ->
                scope.block().also { scope.remove(it.value) }
            }

        @JvmName("TEMP")
inline fun <T : IOType> IOScope.launch(block: IOScope.() -> T): T = IOScope()
            .use { scope ->
                scope.block()
                    .also { scope.remove(it.value) }
                    .also { this.register(it.value) }
            }
    }
}
