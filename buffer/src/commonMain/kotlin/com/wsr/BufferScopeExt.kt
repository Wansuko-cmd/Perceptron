package com.wsr

import com.wsr.base.data.DataBuffer
import com.wsr.base.data.DataBufferGenerator
import kotlin.jvm.JvmName

context(scope: BufferScope)
fun Backend.plus(x: Float, y: DataBuffer): DataBuffer = plus(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(x: DataBuffer, y: Float): DataBuffer = plus(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(x: DataBuffer, y: DataBuffer): DataBuffer = plus(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer =
    plus(x, y, yi, yj, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer =
    plus(x, y, yi, yj, yk, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer =
    plus(x, xi, xj, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(
    x: DataBuffer,
    xi: Int,
    xj: Int,
    y: DataBuffer,
    yi: Int,
    yj: Int,
    yk: Int,
    axis1: Int,
    axis2: Int,
): DataBuffer = plus(x, xi, xj, y, yi, yj, yk, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer =
    plus(x, xi, xj, xk, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(
    x: DataBuffer,
    xi: Int,
    xj: Int,
    xk: Int,
    y: DataBuffer,
    yi: Int,
    yj: Int,
    axis1: Int,
    axis2: Int,
): DataBuffer = plus(x, xi, xj, xk, y, yi, yj, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(
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
): DataBuffer = plus(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer =
    plus(x, xi, xj, xk, xl, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(
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
): DataBuffer = plus(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.plus(
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
): DataBuffer = plus(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(x: Float, y: DataBuffer): DataBuffer = minus(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(x: DataBuffer, y: Float): DataBuffer = minus(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(x: DataBuffer, y: DataBuffer): DataBuffer = minus(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer =
    minus(x, y, yi, yj, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer =
    minus(x, y, yi, yj, yk, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer =
    minus(x, xi, xj, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(
    x: DataBuffer,
    xi: Int,
    xj: Int,
    y: DataBuffer,
    yi: Int,
    yj: Int,
    yk: Int,
    axis1: Int,
    axis2: Int,
): DataBuffer = minus(x, xi, xj, y, yi, yj, yk, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer =
    minus(x, xi, xj, xk, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(
    x: DataBuffer,
    xi: Int,
    xj: Int,
    xk: Int,
    y: DataBuffer,
    yi: Int,
    yj: Int,
    axis1: Int,
    axis2: Int,
): DataBuffer = minus(x, xi, xj, xk, y, yi, yj, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(
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
): DataBuffer = minus(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer =
    minus(x, xi, xj, xk, xl, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(
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
): DataBuffer = minus(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.minus(
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
): DataBuffer = minus(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(x: Float, y: DataBuffer): DataBuffer = times(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(x: DataBuffer, y: Float): DataBuffer = times(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(x: DataBuffer, y: DataBuffer): DataBuffer = times(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer =
    times(x, y, yi, yj, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer =
    times(x, y, yi, yj, yk, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer =
    times(x, xi, xj, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(
    x: DataBuffer,
    xi: Int,
    xj: Int,
    y: DataBuffer,
    yi: Int,
    yj: Int,
    yk: Int,
    axis1: Int,
    axis2: Int,
): DataBuffer = times(x, xi, xj, y, yi, yj, yk, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer =
    times(x, xi, xj, xk, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(
    x: DataBuffer,
    xi: Int,
    xj: Int,
    xk: Int,
    y: DataBuffer,
    yi: Int,
    yj: Int,
    axis1: Int,
    axis2: Int,
): DataBuffer = times(x, xi, xj, xk, y, yi, yj, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(
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
): DataBuffer = times(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer =
    times(x, xi, xj, xk, xl, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(
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
): DataBuffer = times(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.times(
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
): DataBuffer = times(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(x: Float, y: DataBuffer): DataBuffer = div(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(x: DataBuffer, y: Float): DataBuffer = div(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(x: DataBuffer, y: DataBuffer): DataBuffer = div(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer =
    div(x, y, yi, yj, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer =
    div(x, y, yi, yj, yk, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer =
    div(x, xi, xj, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(
    x: DataBuffer,
    xi: Int,
    xj: Int,
    y: DataBuffer,
    yi: Int,
    yj: Int,
    yk: Int,
    axis1: Int,
    axis2: Int,
): DataBuffer = div(x, xi, xj, y, yi, yj, yk, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer =
    div(x, xi, xj, xk, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(
    x: DataBuffer,
    xi: Int,
    xj: Int,
    xk: Int,
    y: DataBuffer,
    yi: Int,
    yj: Int,
    axis1: Int,
    axis2: Int,
): DataBuffer = div(x, xi, xj, xk, y, yi, yj, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(
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
): DataBuffer = div(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer =
    div(x, xi, xj, xk, xl, y, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(
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
): DataBuffer = div(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.div(
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
): DataBuffer = div(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.inner(x: DataBuffer, y: DataBuffer, b: Int): DataBuffer = inner(x, y, b).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.matMul(x: DataBuffer, y: DataBuffer, transY: Boolean, n: Int, k: Int): DataBuffer =
    matMul(x, y, transY, n, k).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.matMul(x: DataBuffer, transX: Boolean, y: DataBuffer, m: Int, k: Int): DataBuffer =
    matMul(x, transX, y, m, k).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.matMul(
    x: DataBuffer,
    transX: Boolean,
    y: DataBuffer,
    transY: Boolean,
    m: Int,
    n: Int,
    k: Int,
    b: Int,
): DataBuffer = matMul(x, transX, y, transY, m, n, k, b).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.exp(x: DataBuffer): DataBuffer = exp(x).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.ln(x: DataBuffer, e: Float): DataBuffer = ln(x, e).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.sigmoid(x: DataBuffer): DataBuffer = sigmoid(x).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.pow(x: DataBuffer, n: Int): DataBuffer = pow(x, n).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.sqrt(x: DataBuffer, e: Float): DataBuffer = sqrt(x, e).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.average(x: DataBuffer): DataBuffer = average(x).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.average(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer =
    average(x, xi, xj, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.average(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer =
    average(x, xi, xj, xk, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.max(x: DataBuffer): DataBuffer = max(x).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.max(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer =
    max(x, xi, xj, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.max(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer =
    max(x, xi, xj, xk, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.min(x: DataBuffer): DataBuffer = min(x).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.min(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer =
    min(x, xi, xj, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.min(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer =
    min(x, xi, xj, xk, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.sum(x: DataBuffer): DataBuffer = sum(x).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.sum(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer =
    sum(x, xi, xj, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.sum(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer =
    sum(x, xi, xj, xk, axis).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.transpose(x: DataBuffer, xi: Int, xj: Int): DataBuffer = transpose(x, xi, xj).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.transpose(x: DataBuffer, xi: Int, xj: Int, xk: Int, axisI: Int, axisJ: Int, axisK: Int): DataBuffer =
    transpose(x, xi, xj, xk, axisI, axisJ, axisK).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.transpose(
    x: DataBuffer,
    xi: Int,
    xj: Int,
    xk: Int,
    xl: Int,
    axisI: Int,
    axisJ: Int,
    axisK: Int,
    axisL: Int,
): DataBuffer = transpose(x, xi, xj, xk, xl, axisI, axisJ, axisK, axisL).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.slice(x: DataBuffer, indices: IntProgression): DataBuffer = slice(x, indices).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.slice(x: DataBuffer, xi: Int, xj: Int, axis: Int, indices: IntProgression): DataBuffer =
    slice(x, xi, xj, axis, indices).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.slice(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int, indices: IntProgression): DataBuffer =
    slice(x, xi, xj, xk, axis, indices).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.gather(x: DataBuffer, y: DataBuffer, i: Int, j: Int, k: Int): DataBuffer =
    gather(x, y, i, j, k).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.scatterAdd(x: DataBuffer, y: DataBuffer, i: Int, j: Int, k: Int, b: Int): DataBuffer =
    scatterAdd(x, y, i, j, k, b).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.greaterThan(x: DataBuffer, y: Float): DataBuffer = greaterThan(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.greaterThan(x: DataBuffer, y: DataBuffer): DataBuffer = greaterThan(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.lessThan(x: DataBuffer, y: Float): DataBuffer = lessThan(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.lessThan(x: DataBuffer, y: DataBuffer): DataBuffer = lessThan(x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.equals(x: DataBuffer, y: Float, absoluteTolerance: Float, relativeTolerance: Float): DataBuffer =
    equals(x, y, absoluteTolerance, relativeTolerance).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.equals(x: DataBuffer, y: DataBuffer, absoluteTolerance: Float, relativeTolerance: Float): DataBuffer =
    equals(x, y, absoluteTolerance, relativeTolerance).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.where(condition: DataBuffer, x: Float, y: Float): DataBuffer =
    where(condition, x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.where(condition: DataBuffer, x: Float, y: DataBuffer): DataBuffer =
    where(condition, x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.where(condition: DataBuffer, x: DataBuffer, y: Float): DataBuffer =
    where(condition, x, y).also { scope.register(it) }

context(scope: BufferScope)
fun Backend.where(condition: DataBuffer, x: DataBuffer, y: DataBuffer): DataBuffer =
    where(condition, x, y).also { scope.register(it) }

context(scope: BufferScope)
fun create(size: Int) = DataBufferGenerator.create(size).also { scope.register(it) }

context(scope: BufferScope)
fun create(value: FloatArray) = DataBufferGenerator.create(value).also { scope.register(it) }

@JvmName("createWithElements")
context(scope: BufferScope)
fun create(vararg elements: Float) = DataBufferGenerator.create(elements).also { scope.register(it) }
