package com.wsr.knist

import com.wsr.knist.base.IBackend
import com.wsr.knist.base.data.DataBuffer
import com.wsr.knist.base.data.DataBufferGenerator
import com.wsr.knist.base.data.IDataBufferGenerator
import com.wsr.knist.cpu.cpu
import kotlin.random.Random

object Backend : IBackend {
    private var instance: IBackend = cpu

    init {
        DataBufferGenerator.set(instance.generator)
    }

    fun set(backend: IBackend) {
        instance = backend
        DataBufferGenerator.set(backend.generator)
    }

    override val generator: IDataBufferGenerator get() = instance.generator

    override fun plus(x: Float, y: DataBuffer): DataBuffer = instance.plus(x, y)

    override fun plus(x: DataBuffer, y: Float): DataBuffer = instance.plus(x, y)

    override fun plus(x: DataBuffer, y: DataBuffer): DataBuffer = instance.plus(x, y)

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer =
        instance.plus(x, y, yi, yj, axis)

    override fun plus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer =
        instance.plus(x, y, yi, yj, yk, axis)

    override fun plus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.plus(x, xi, xj, y, axis)

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
    ): DataBuffer = instance.plus(x, xi, xj, y, yi, yj, yk, axis1, axis2)

    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.plus(x, xi, xj, xk, y, axis)

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
    ): DataBuffer = instance.plus(x, xi, xj, xk, y, yi, yj, axis1, axis2)

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
    ): DataBuffer = instance.plus(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3)

    override fun plus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.plus(x, xi, xj, xk, xl, y, axis)

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
    ): DataBuffer = instance.plus(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2)

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
    ): DataBuffer = instance.plus(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3)

    override fun minus(x: Float, y: DataBuffer): DataBuffer = instance.minus(x, y)

    override fun minus(x: DataBuffer, y: Float): DataBuffer = instance.minus(x, y)

    override fun minus(x: DataBuffer, y: DataBuffer): DataBuffer = instance.minus(x, y)

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer =
        instance.minus(x, y, yi, yj, axis)

    override fun minus(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer =
        instance.minus(x, y, yi, yj, yk, axis)

    override fun minus(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.minus(x, xi, xj, y, axis)

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
    ): DataBuffer = instance.minus(x, xi, xj, y, yi, yj, yk, axis1, axis2)

    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.minus(x, xi, xj, xk, y, axis)

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
    ): DataBuffer = instance.minus(x, xi, xj, xk, y, yi, yj, axis1, axis2)

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
    ): DataBuffer = instance.minus(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3)

    override fun minus(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.minus(x, xi, xj, xk, xl, y, axis)

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
    ): DataBuffer = instance.minus(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2)

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
    ): DataBuffer = instance.minus(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3)

    override fun times(x: Float, y: DataBuffer): DataBuffer = instance.times(x, y)

    override fun times(x: DataBuffer, y: Float): DataBuffer = instance.times(x, y)

    override fun times(x: DataBuffer, y: DataBuffer): DataBuffer = instance.times(x, y)

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer =
        instance.times(x, y, yi, yj, axis)

    override fun times(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer =
        instance.times(x, y, yi, yj, yk, axis)

    override fun times(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.times(x, xi, xj, y, axis)

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
    ): DataBuffer = instance.times(x, xi, xj, y, yi, yj, yk, axis1, axis2)

    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.times(x, xi, xj, xk, y, axis)

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
    ): DataBuffer = instance.times(x, xi, xj, xk, y, yi, yj, axis1, axis2)

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
    ): DataBuffer = instance.times(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3)

    override fun times(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.times(x, xi, xj, xk, xl, y, axis)

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
    ): DataBuffer = instance.times(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2)

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
    ): DataBuffer = instance.times(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3)

    override fun div(x: Float, y: DataBuffer): DataBuffer = instance.div(x, y)

    override fun div(x: DataBuffer, y: Float): DataBuffer = instance.div(x, y)

    override fun div(x: DataBuffer, y: DataBuffer): DataBuffer = instance.div(x, y)

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int): DataBuffer =
        instance.div(x, y, yi, yj, axis)

    override fun div(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int): DataBuffer =
        instance.div(x, y, yi, yj, yk, axis)

    override fun div(x: DataBuffer, xi: Int, xj: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.div(x, xi, xj, y, axis)

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
    ): DataBuffer = instance.div(x, xi, xj, y, yi, yj, yk, axis1, axis2)

    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.div(x, xi, xj, xk, y, axis)

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
    ): DataBuffer = instance.div(x, xi, xj, xk, y, yi, yj, axis1, axis2)

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
    ): DataBuffer = instance.div(x, xi, xj, xk, y, yi, yj, yk, yl, axis1, axis2, axis3)

    override fun div(x: DataBuffer, xi: Int, xj: Int, xk: Int, xl: Int, y: DataBuffer, axis: Int): DataBuffer =
        instance.div(x, xi, xj, xk, xl, y, axis)

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
    ): DataBuffer = instance.div(x, xi, xj, xk, xl, y, yi, yj, axis1, axis2)

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
    ): DataBuffer = instance.div(x, xi, xj, xk, xl, y, yi, yj, yk, axis1, axis2, axis3)

    override fun inner(x: DataBuffer, y: DataBuffer, b: Int): DataBuffer = instance.inner(x, y, b)

    override fun matMul(x: DataBuffer, y: DataBuffer, transY: Boolean, n: Int, k: Int): DataBuffer =
        instance.matMul(x, y, transY, n, k)

    override fun matMul(x: DataBuffer, transX: Boolean, y: DataBuffer, m: Int, k: Int): DataBuffer =
        instance.matMul(x, transX, y, m, k)

    override fun matMul(
        x: DataBuffer,
        transX: Boolean,
        y: DataBuffer,
        transY: Boolean,
        m: Int,
        n: Int,
        k: Int,
        b: Int,
    ): DataBuffer = instance.matMul(x, transX, y, transY, m, n, k, b)

    override fun exp(x: DataBuffer): DataBuffer = instance.exp(x)

    override fun ln(x: DataBuffer, e: Float): DataBuffer = instance.ln(x, e)

    override fun sigmoid(x: DataBuffer): DataBuffer = instance.sigmoid(x)

    override fun pow(x: DataBuffer, n: Int): DataBuffer = instance.pow(x, n)

    override fun sqrt(x: DataBuffer, e: Float): DataBuffer = instance.sqrt(x, e)

    override fun random(size: Int, from: Float, until: Float, random: Random): DataBuffer =
        instance.random(size, from, until, random)

    override fun average(x: DataBuffer): DataBuffer = instance.average(x)

    override fun average(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer = instance.average(x, xi, xj, axis)

    override fun average(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer =
        instance.average(x, xi, xj, xk, axis)

    override fun max(x: DataBuffer): DataBuffer = instance.max(x)

    override fun max(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer = instance.max(x, xi, xj, axis)

    override fun max(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer =
        instance.max(x, xi, xj, xk, axis)

    override fun min(x: DataBuffer): DataBuffer = instance.min(x)

    override fun min(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer = instance.min(x, xi, xj, axis)

    override fun min(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer =
        instance.min(x, xi, xj, xk, axis)

    override fun sum(x: DataBuffer): DataBuffer = instance.sum(x)

    override fun sum(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer = instance.sum(x, xi, xj, axis)

    override fun sum(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer =
        instance.sum(x, xi, xj, xk, axis)

    override fun maxIndex(x: DataBuffer): DataBuffer = instance.maxIndex(x)

    override fun maxIndex(x: DataBuffer, xi: Int, xj: Int, axis: Int): DataBuffer = instance.maxIndex(x, xi, xj, axis)

    override fun maxIndex(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer =
        instance.maxIndex(x, xi, xj, xk, axis)

    override fun topK(x: DataBuffer, k: Int, random: Random): DataBuffer = instance.topK(x, k, random)

    override fun topK(x: DataBuffer, xi: Int, xj: Int, k: Int, axis: Int, random: Random): DataBuffer =
        instance.topK(x, xi, xj, k, axis, random)

    override fun topK(x: DataBuffer, xi: Int, xj: Int, xk: Int, k: Int, axis: Int, random: Random): DataBuffer =
        instance.topK(x, xi, xj, xk, k, axis, random)

    override fun topP(x: DataBuffer, p: Float, random: Random): DataBuffer = instance.topP(x, p, random)

    override fun topP(x: DataBuffer, xi: Int, xj: Int, p: Float, axis: Int, random: Random): DataBuffer =
        instance.topP(x, xi, xj, p, axis, random)

    override fun topP(x: DataBuffer, xi: Int, xj: Int, xk: Int, p: Float, axis: Int, random: Random): DataBuffer =
        instance.topP(x, xi, xj, xk, p, axis, random)

    override fun transpose(x: DataBuffer, xi: Int, xj: Int): DataBuffer = instance.transpose(x, xi, xj)

    override fun transpose(x: DataBuffer, xi: Int, xj: Int, xk: Int, axisI: Int, axisJ: Int, axisK: Int): DataBuffer =
        instance.transpose(x, xi, xj, xk, axisI, axisJ, axisK)

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
    ): DataBuffer = instance.transpose(x, xi, xj, xk, xl, axisI, axisJ, axisK, axisL)

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
    ): DataBuffer = instance.transpose(x, xi, xj, xk, xl, xm, axisI, axisJ, axisK, axisL, axisM)

    override fun slice(x: DataBuffer, indices: IntProgression): DataBuffer = instance.slice(x, indices)

    override fun slice(x: DataBuffer, xi: Int, xj: Int, axis: Int, indices: IntProgression): DataBuffer =
        instance.slice(x, xi, xj, axis, indices)

    override fun slice(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int, indices: IntProgression): DataBuffer =
        instance.slice(x, xi, xj, xk, axis, indices)

    override fun copyInto(x: DataBuffer, y: DataBuffer, indices: IntProgression) {
        instance.copyInto(x, y, indices)
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, axis: Int, indices: IntProgression) {
        instance.copyInto(x, y, yi, yj, axis, indices)
    }

    override fun copyInto(x: DataBuffer, y: DataBuffer, yi: Int, yj: Int, yk: Int, axis: Int, indices: IntProgression) {
        instance.copyInto(x, y, yi, yj, yk, axis, indices)
    }

    override fun gather(x: DataBuffer, y: DataBuffer, i: Int, j: Int, k: Int): DataBuffer =
        instance.gather(x, y, i, j, k)

    override fun scatterAdd(x: DataBuffer, y: DataBuffer, i: Int, j: Int, k: Int, b: Int): DataBuffer =
        instance.scatterAdd(x, y, i, j, k, b)

    override fun greaterThan(x: DataBuffer, y: Float): DataBuffer = instance.greaterThan(x, y)

    override fun greaterThan(x: DataBuffer, y: DataBuffer): DataBuffer = instance.greaterThan(x, y)

    override fun lessThan(x: DataBuffer, y: Float): DataBuffer = instance.lessThan(x, y)

    override fun lessThan(x: DataBuffer, y: DataBuffer): DataBuffer = instance.lessThan(x, y)

    override fun equals(x: DataBuffer, y: Float, absoluteTolerance: Float, relativeTolerance: Float): DataBuffer =
        instance.equals(x, y, absoluteTolerance, relativeTolerance)

    override fun equals(x: DataBuffer, y: DataBuffer, absoluteTolerance: Float, relativeTolerance: Float): DataBuffer =
        instance.equals(x, y, absoluteTolerance, relativeTolerance)

    override fun where(condition: DataBuffer, x: Float, y: Float): DataBuffer = instance.where(condition, x, y)

    override fun where(condition: DataBuffer, x: Float, y: DataBuffer): DataBuffer = instance.where(condition, x, y)

    override fun where(condition: DataBuffer, x: DataBuffer, y: Float): DataBuffer = instance.where(condition, x, y)

    override fun where(condition: DataBuffer, x: DataBuffer, y: DataBuffer): DataBuffer =
        instance.where(condition, x, y)

    override fun unfold(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        b: Int,
        window: Int,
        stride: Int,
        dilation: Int,
        padding: Int,
    ): DataBuffer = instance.unfold(x, xi, xj, b, window, stride, dilation, padding)

    override fun unfold(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        b: Int,
        window: Int,
        stride: Int,
        dilation: Int,
        padding: Int,
    ): DataBuffer = instance.unfold(x, xi, xj, xk, b, window, stride, dilation, padding)

    override fun fold(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        b: Int,
        stride: Int,
        dilation: Int,
        padding: Int,
    ): DataBuffer = instance.fold(x, xi, xj, xk, b, stride, dilation, padding)

    override fun fold(
        x: DataBuffer,
        xi: Int,
        xj: Int,
        xk: Int,
        xl: Int,
        b: Int,
        stride: Int,
        dilation: Int,
        padding: Int,
    ): DataBuffer = instance.fold(x, xi, xj, xk, xl, b, stride, dilation, padding)

    override fun flip(x: DataBuffer, xi: Int, xj: Int, xk: Int, axis: Int): DataBuffer =
        instance.flip(x, xi, xj, xk, axis)

    override fun flush() = instance.flush()
    override fun sync() = instance.sync()
}
